"""Render walk, trot, and bound rollouts at the documented fixed speeds."""

from __future__ import annotations

import os
import sys

os.environ.setdefault(
    "MUJOCO_GL", "glfw" if sys.platform == "darwin" else "osmesa"
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/barkour-matplotlib")

import argparse
import csv
import json
import math as python_math
from pathlib import Path

import jax
from jax import numpy as jp
import mediapy as media
import numpy as np
from brax import math
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from barkour import BarkourEnv
from coeff_config import (
    EARLY_TERMINATION_PENALTY,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    TRAINING_COMMAND_SAMPLING,
    VISUAL_EVAL_SPEEDS,
    WALK_TO_TROT_ENTER,
    eps_vy,
    eps_wz,
)


POLICY_HIDDEN_LAYER_SIZES = (128, 128, 128, 128)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument(
      "--output_dir", type=Path, default=Path("visual_evaluation")
  )
  parser.add_argument(
      "--speeds", type=float, nargs="+", default=VISUAL_EVAL_SPEEDS
  )
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--warmup_steps", type=int, default=50)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--render_every", type=int, default=2)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument(
      "--no_render",
      action="store_true",
      help="Write numeric traces and summaries without videos or contact sheets.",
  )
  return parser.parse_args()


def reset_with_command(env: BarkourEnv, rng: jax.Array, vx: float):
  """Reset all command-conditioned state and force a forward command."""
  state = env.reset(rng)
  info = dict(state.info)
  command = jp.array([vx, 0.0, 0.0], dtype=jp.float32)
  info["command"] = command
  info["mode"] = env._mode_from_command(command)
  info["step"] = jp.array(0, dtype=jp.int32)
  info["history_len"] = jp.array(0, dtype=jp.int32)
  for key, value in tuple(info.items()):
    if key.endswith("_history"):
      info[key] = jp.zeros_like(value)
  for key in ("last_act", "last_vel", "last_contact", "feet_air_time", "kick"):
    info[key] = jp.zeros_like(info[key])
  obs = env._get_obs(state.pipeline_state, info, jp.zeros_like(state.obs))
  return state.replace(obs=obs, info=info)


def write_contact_sheet(
    frames: list[np.ndarray], path: Path, max_frames: int = 12, columns: int = 4
) -> None:
  """Write evenly spaced rollout frames as one image for quick visual QA."""
  if not frames:
    raise ValueError("Cannot make a contact sheet from an empty frame list.")
  count = min(max_frames, len(frames))
  indices = np.linspace(0, len(frames) - 1, count, dtype=np.int32)
  sampled = [np.asarray(frames[index]) for index in indices]
  frame_height, frame_width = sampled[0].shape[:2]
  channels = sampled[0].shape[2]
  rows = python_math.ceil(count / columns)
  sheet = np.zeros(
      (rows * frame_height, columns * frame_width, channels),
      dtype=sampled[0].dtype,
  )
  for index, frame in enumerate(sampled):
    row, column = divmod(index, columns)
    sheet[
        row * frame_height : (row + 1) * frame_height,
        column * frame_width : (column + 1) * frame_width,
    ] = frame
  media.write_image(path, sheet)


def metric_value(state, name: str) -> float:
  return float(np.asarray(state.metrics[name]))


def main() -> None:
  args = parse_args()
  if args.steps <= 0 or args.render_every <= 0:
    raise ValueError("--steps and --render_every must be positive.")
  if args.warmup_steps < 0:
    raise ValueError("--warmup_steps must be nonnegative.")
  if tuple(float(value) for value in args.speeds) != tuple(VISUAL_EVAL_SPEEDS):
    raise ValueError(
        "This evaluation must use the documented walk/trot/bound speed set: "
        f"{VISUAL_EVAL_SPEEDS}."
    )

  policy_path = args.policy.expanduser().resolve()
  if not policy_path.is_file():
    raise FileNotFoundError(f"Policy file does not exist: {policy_path}")
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  env = BarkourEnv(
      obs_noise=0.0,
      kick_vel=0.0,
      command_sampling=TRAINING_COMMAND_SAMPLING,
      early_termination_penalty=EARLY_TERMINATION_PENALTY,
  )
  network = ppo_networks.make_ppo_networks(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
  )
  params = model.load_params(policy_path)
  inference_fn = jax.jit(
      ppo_networks.make_inference_fn(network)(params, deterministic=True)
  )
  step_fn = jax.jit(env.step)

  summaries = []
  for speed_index, speed in enumerate(args.speeds):
    rng = jax.random.PRNGKey(args.seed + speed_index)
    state = reset_with_command(env, rng, speed)
    rollout = [state.pipeline_state]
    rows = []
    first_done_step = None

    for step in range(args.steps):
      mode_before_step = int(np.asarray(state.info["mode"]))
      rng, action_rng = jax.random.split(rng)
      action, _ = inference_fn(state.obs, action_rng)
      state = step_fn(state, action)
      rollout.append(state.pipeline_state)

      torso_index = env._torso_idx - 1
      local_velocity = math.rotate(
          state.pipeline_state.xd.vel[torso_index],
          math.quat_inv(state.pipeline_state.x.rot[torso_index]),
      )
      local_angular_velocity = math.rotate(
          state.pipeline_state.xd.ang[torso_index],
          math.quat_inv(state.pipeline_state.x.rot[torso_index]),
      )
      contacts = np.asarray(state.info["last_contact"], dtype=bool)
      done = bool(np.asarray(state.done))
      if done and first_done_step is None:
        first_done_step = step + 1

      rows.append(
          {
              "step": step + 1,
              "command_vx": speed,
              "mode": mode_before_step,
              "actual_vx": float(np.asarray(local_velocity[0])),
              "actual_vy": float(np.asarray(local_velocity[1])),
              "actual_yaw_rate": float(np.asarray(local_angular_velocity[2])),
              "center_of_mass_z": float(
                  np.asarray(state.pipeline_state.subtree_com[0, 2])
              ),
              "torso_height": float(
                  np.asarray(state.pipeline_state.x.pos[torso_index, 2])
              ),
              "reward": float(np.asarray(state.reward)),
              "done": int(done),
              "contact_front_left": int(contacts[0]),
              "contact_hind_left": int(contacts[1]),
              "contact_front_right": int(contacts[2]),
              "contact_hind_right": int(contacts[3]),
              "rho_stl": metric_value(state, "rho_stl"),
              "rho_stl_smooth": metric_value(state, "rho_stl_smooth"),
              "rho_tracking": metric_value(state, "rho_tracking"),
              "rho_safety": metric_value(state, "rho_safety"),
              "rho_contact": metric_value(state, "rho_contact"),
              "rho_com_z": metric_value(state, "rho_com_z"),
              "rho_tilt": metric_value(state, "rho_tilt"),
          }
      )
      if done:
        break

    stem = f"vx_{speed:.1f}"
    trace_path = output_dir / f"{stem}.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=rows[0])
      writer.writeheader()
      writer.writerows(rows)

    video_path = None
    contact_sheet_path = None
    if not args.no_render:
      frames = env.render(
          rollout[:: args.render_every],
          camera="track",
          width=args.width,
          height=args.height,
      )
      video_path = output_dir / f"{stem}.mp4"
      media.write_video(
          video_path,
          frames,
          fps=1.0 / env.dt / args.render_every,
      )
      contact_sheet_path = output_dir / f"{stem}_contact_sheet.png"
      write_contact_sheet(frames, contact_sheet_path)

    analysis_rows = rows[args.warmup_steps :]
    if not analysis_rows:
      analysis_rows = rows
    actual_vx = np.asarray(
        [row["actual_vx"] for row in analysis_rows], dtype=np.float32
    )
    actual_vy = np.asarray(
        [row["actual_vy"] for row in analysis_rows], dtype=np.float32
    )
    actual_yaw_rate = np.asarray(
        [row["actual_yaw_rate"] for row in analysis_rows], dtype=np.float32
    )
    com_z = np.asarray(
        [row["center_of_mass_z"] for row in rows], dtype=np.float32
    )
    rho_stl = np.asarray(
        [row["rho_stl"] for row in analysis_rows], dtype=np.float32
    )
    mean_actual_vx = float(actual_vx.mean())
    mean_absolute_vy = float(np.abs(actual_vy).mean())
    mean_absolute_yaw_rate = float(np.abs(actual_yaw_rate).mean())
    net_lateral_displacement_proxy = float(np.sum(actual_vy) * env.dt)
    net_heading_change_degrees = float(
        np.degrees(np.sum(actual_yaw_rate) * env.dt)
    )
    accumulated_absolute_heading_change_degrees = float(
        np.degrees(np.sum(np.abs(actual_yaw_rate)) * env.dt)
    )
    tracking_tolerance = 0.15 * abs(float(speed))
    if speed >= TROT_TO_BOUND_ENTER:
      expected_mode_id = MODE_BOUND
      expected_mode_name = "bound"
    elif speed >= WALK_TO_TROT_ENTER:
      expected_mode_id = MODE_TROT
      expected_mode_name = "trot"
    else:
      expected_mode_id = MODE_WALK
      expected_mode_name = "walk"
    modes = sorted({row["mode"] for row in rows})
    survived = first_done_step is None and len(rows) == args.steps
    summaries.append(
        {
            "command_vx": speed,
            "expected_mode": expected_mode_name,
            "expected_mode_id": expected_mode_id,
            "observed_mode_ids": modes,
            "mode_check_passed": modes == [expected_mode_id],
            "survived_full_horizon": survived,
            "survived_steps": len(rows),
            "first_done_step": first_done_step,
            "mean_actual_vx_after_warmup": mean_actual_vx,
            "mean_absolute_velocity_error_after_warmup": float(
                np.abs(actual_vx - speed).mean()
            ),
            "mean_absolute_lateral_velocity_after_warmup": mean_absolute_vy,
            "lateral_tracking_mean_within_eps_vy": bool(
                mean_absolute_vy <= eps_vy
            ),
            "net_lateral_displacement_proxy_after_warmup_m": (
                net_lateral_displacement_proxy
            ),
            "mean_absolute_yaw_rate_after_warmup": mean_absolute_yaw_rate,
            "yaw_tracking_mean_within_eps_wz": bool(
                mean_absolute_yaw_rate <= eps_wz
            ),
            "net_heading_change_after_warmup_degrees": (
                net_heading_change_degrees
            ),
            "accumulated_absolute_heading_change_after_warmup_degrees": (
                accumulated_absolute_heading_change_degrees
            ),
            "mean_velocity_within_15_percent": bool(
                abs(mean_actual_vx - speed) <= tracking_tolerance
            ),
            "minimum_center_of_mass_z": float(com_z.min()),
            "stl_satisfaction_fraction_after_warmup": float(
                np.mean(rho_stl >= 0.0)
            ),
            "final_x": float(
                np.asarray(state.pipeline_state.x.pos[env._torso_idx - 1, 0])
            ),
            "video": str(video_path) if video_path is not None else None,
            "contact_sheet": (
                str(contact_sheet_path)
                if contact_sheet_path is not None
                else None
            ),
            "trace": str(trace_path),
        }
    )

  summary_path = output_dir / "summary.json"
  summary_path.write_text(
      json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  main()
