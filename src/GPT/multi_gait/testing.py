"""Render deterministic fixed-speed rollouts from a trained Barkour policy."""

from __future__ import annotations

import os
import sys
os.environ.setdefault(
    "MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl"
)

import argparse
import csv
import json
from pathlib import Path

import jax
from jax import numpy as jp
import mediapy as media
import numpy as np
from brax import math
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from coeff_config import (
    EARLY_TERMINATION_PENALTY,
    TRAINING_COMMAND_SAMPLING,
    VISUAL_EVAL_SPEEDS,
)
from velocity_tracking import (
    command_relative_weighted_tracking_error,
    instantaneous_tracking_robustness,
)


def tracking_diagnostics_for_evaluation(
    command,
    body_linear_velocity,
    body_yaw_rate,
):
  """Returns the same tracking scalar used by the training environment."""
  result = command_relative_weighted_tracking_error(
      command=command,
      body_linear_velocity=body_linear_velocity,
      body_yaw_rate=body_yaw_rate,
  )
  return {
      "forward_normalized_error": result.forward_normalized_error,
      "lateral_normalized_error": result.lateral_normalized_error,
      "yaw_normalized_error": result.yaw_normalized_error,
      "velocity_error": result.velocity_error,
      "rho_tracking_atomic": instantaneous_tracking_robustness(
          result.velocity_error
      ),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, required=True)
  parser.add_argument(
      "--output_dir", type=Path, default=Path("visual_evaluation")
  )
  parser.add_argument(
      "--speeds",
      type=float,
      nargs="+",
      default=VISUAL_EVAL_SPEEDS,
  )
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--selection_quantile",
      choices=("q50",),
      default="q50",
      help="Frozen q50 STL predicate set used by this treatment.",
  )
  parser.add_argument("--render_every", type=int, default=2)
  parser.add_argument(
      "--no-render",
      action="store_true",
      help="Write deterministic traces and summaries without MP4 rendering.",
  )
  return parser.parse_args()


def reset_with_command(env, rng: jax.Array, vx: float):
  """Resets the environment and replaces the sampled command with fixed vx."""
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
  for key in (
      "last_act",
      "last_vel",
      "last_contact",
      "feet_air_time",
      "kick",
  ):
    info[key] = jp.zeros_like(info[key])
  obs = env._get_obs(state.pipeline_state, info, jp.zeros_like(state.obs))
  return state.replace(obs=obs, info=info)


def main() -> None:
  args = parse_args()
  os.environ["STL_SELECTION_QUANTILE"] = args.selection_quantile
  from barkour import BarkourEnv

  policy_path = args.policy.expanduser().resolve()
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
      policy_hidden_layer_sizes=(128, 128, 128, 128),
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
      tracking = tracking_diagnostics_for_evaluation(
          command=jp.array([speed, 0.0, 0.0], dtype=jp.float32),
          body_linear_velocity=local_velocity,
          body_yaw_rate=local_angular_velocity[2],
      )
      contacts = np.asarray(state.info["last_contact"], dtype=bool)
      done = bool(np.asarray(state.done))
      if done and first_done_step is None:
        first_done_step = step + 1
      rows.append({
          "step": step + 1,
          "command_vx": speed,
          # A terminal step resamples the next episode's command and mode.
          # Record the mode that actually produced this transition.
          "mode": mode_before_step,
          "actual_vx": float(np.asarray(local_velocity[0])),
          "actual_vy": float(np.asarray(local_velocity[1])),
          "actual_yaw_rate": float(np.asarray(local_angular_velocity[2])),
          "forward_normalized_error": float(
              np.asarray(tracking["forward_normalized_error"])
          ),
          "lateral_normalized_error": float(
              np.asarray(tracking["lateral_normalized_error"])
          ),
          "yaw_normalized_error": float(
              np.asarray(tracking["yaw_normalized_error"])
          ),
          "velocity_error": float(np.asarray(tracking["velocity_error"])),
          "rho_tracking_atomic": float(
              np.asarray(tracking["rho_tracking_atomic"])
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
          "rho_safety": float(np.asarray(state.metrics["rho_safety"])),
          "rho_tracking": float(np.asarray(state.metrics["rho_tracking"])),
          "rho_tracking_raw": float(
              np.asarray(state.metrics["rho_tracking_raw"])
          ),
          "rho_gait": float(np.asarray(state.metrics["rho_gait"])),
      })
      if done:
        break

    stem = f"vx_{speed:.1f}"
    with (output_dir / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
      writer = csv.DictWriter(handle, fieldnames=rows[0])
      writer.writeheader()
      writer.writerows(rows)

    video_path = None
    if not args.no_render:
      frames = env.render(
          rollout[:: args.render_every],
          camera="track",
          width=640,
          height=480,
      )
      video_path = output_dir / f"{stem}.mp4"
      media.write_video(
          video_path,
          frames,
          fps=1.0 / env.dt / args.render_every,
      )

    analysis_rows = rows[50:] if len(rows) > 50 else rows
    actual_vx = np.asarray(
        [row["actual_vx"] for row in analysis_rows], dtype=np.float32
    )
    actual_vy = np.asarray(
        [row["actual_vy"] for row in analysis_rows], dtype=np.float32
    )
    actual_yaw_rate = np.asarray(
        [row["actual_yaw_rate"] for row in analysis_rows], dtype=np.float32
    )
    weighted_velocity_error = np.asarray(
        [row["velocity_error"] for row in analysis_rows], dtype=np.float32
    )
    heights = np.asarray(
        [row["torso_height"] for row in rows], dtype=np.float32
    )
    rewards = np.asarray(
        [row["reward"] for row in analysis_rows], dtype=np.float32
    )
    modes = sorted({row["mode"] for row in rows})
    summaries.append({
        "command_vx": speed,
        "expected_mode": "walk" if speed < 0.72 else "trot",
        "observed_mode_ids": modes,
        "survived_steps": len(rows),
        "first_done_step": first_done_step,
        "mean_actual_vx_after_warmup": float(actual_vx.mean()),
        "mean_absolute_velocity_error_after_warmup": float(
            np.abs(actual_vx - speed).mean()
        ),
        "mean_weighted_velocity_error_after_warmup": float(
            weighted_velocity_error.mean()
        ),
        "mean_absolute_body_vy_after_warmup": float(np.abs(actual_vy).mean()),
        "mean_absolute_body_yaw_rate_after_warmup": float(
            np.abs(actual_yaw_rate).mean()
        ),
        "minimum_torso_height": float(heights.min()),
        "mean_reward_after_warmup": float(rewards.mean()),
        "final_x": float(
            np.asarray(state.pipeline_state.x.pos[env._torso_idx - 1, 0])
        ),
        "video": str(video_path) if video_path is not None else None,
        "trace": str(output_dir / f"{stem}.csv"),
    })

  summary_path = output_dir / "summary.json"
  summary_path.write_text(
      json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  main()
