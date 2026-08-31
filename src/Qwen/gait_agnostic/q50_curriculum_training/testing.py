#!/usr/bin/env python3
"""Auditable full-range evaluator for the completed q50 curriculum policy.

The evaluator is intentionally independent of the prior run's evaluator: it
uses only this training's gait-agnostic environment and coefficient modules.  It
evaluates a supplied Orbax final-policy directory on the locked protocol and
writes only into a newly created output directory.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
Q50_SOURCE_DIR = Path(__file__).resolve().parents[1] / "q50_filtered_training"
if not Q50_SOURCE_DIR.is_dir():
    raise FileNotFoundError(f"Required retained reward runtime is absent: {Q50_SOURCE_DIR}")
sys.path.insert(0, str(Q50_SOURCE_DIR))

import jax
from jax import numpy as jp
import numpy as np
from brax import math as brax_math
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from coeff_config import MODE_BOUND, MODE_TROT, MODE_WALK, TROT_TO_BOUND_ENTER, WALK_TO_TROT_ENTER


DEFAULT_VELOCITIES = (0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 1.9, 2.0, 2.1)
NUM_TESTS = 20
HORIZON_STEPS = 500
WARMUP_STEPS = 50
OBS_NOISE = 0.0
KICK_VEL = 0.0
GRAVITY = 9.81
CURRENT_OBSERVATION_METADATA = {
    "observation_layout": "current_with_mode_one_hot",
    "observation_dimension": 510,
    "history_frames": 15,
    "features_per_frame": 34,
    "mode_one_hot_included": True,
}
LEGACY_OBSERVATION_METADATA = {
    "observation_layout": "legacy_without_mode_one_hot",
    "observation_dimension": 465,
    "history_frames": 15,
    "features_per_frame": 31,
    "mode_one_hot_included": False,
}


def _install_running_stats_compat_patch():
    """Accept checkpoints written by Brax versions that serialized std_eps."""
    try:
        from brax.training.acme import running_statistics
    except Exception:
        return lambda: None
    cls = running_statistics.RunningStatisticsState
    if "std_eps" in getattr(cls, "__dataclass_fields__", {}):
        return lambda: None
    original_init = cls.__init__

    def patched_init(self, *args, std_eps=None, **kwargs):
        del std_eps
        return original_init(self, *args, **kwargs)

    cls.__init__ = patched_init

    def restore():
        cls.__init__ = original_init

    return restore


def load_policy(env, checkpoint_path: Path):
    """Restore the gait-agnostic 4x128 PPO final policy."""
    restore_patch = _install_running_stats_compat_patch()
    try:
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=0,
            episode_length=1000,
            normalize_observations=True,
            restore_checkpoint_path=str(checkpoint_path.resolve()),
            network_factory=functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(128, 128, 128, 128),
            ),
        )
    finally:
        restore_patch()
    return jax.jit(make_inference_fn(params))


def _mode_from_command(command):
    vx_abs = jp.abs(command[0])
    return jp.where(
        vx_abs >= TROT_TO_BOUND_ENTER,
        jp.asarray(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx_abs >= WALK_TO_TROT_ENTER,
            jp.asarray(MODE_TROT, dtype=jp.int32),
            jp.asarray(MODE_WALK, dtype=jp.int32),
        ),
    )


def reset_eval_state_with_command(env, rng, command):
    """Reset and build an observation containing the fixed command."""
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = _mode_from_command(command)
    info["history_len"] = jp.asarray(0, dtype=jp.int32)
    info["signal_history"] = env._empty_signal_history()
    obs = env._get_obs(state.pipeline_state, info, jp.zeros_like(state.obs))
    return state.replace(obs=obs, info=info)


class LegacyObsBarkourEnv(BarkourEnv):
    """gait-agnostic environment with the checkpoint's legacy 465-observation view."""

    def _get_obs(self, pipeline_state, state_info, obs_history):
        # This intentionally preserves the established compatibility convention:
        # body 0 supplies local velocity/orientation observation quantities.
        inv_torso_rot = brax_math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = brax_math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        obs = jp.concatenate(
            [
                jp.asarray([local_rpyrate[2]]) * 0.25,
                brax_math.rotate(jp.asarray([0, 0, -1]), inv_torso_rot),
                state_info["command"] * jp.asarray([2.0, 2.0, 0.25]),
                pipeline_state.q[7:] - self._default_pose,
                state_info["last_act"],
            ]
        )
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        return jp.roll(obs_history, obs.shape[0]).at[: obs.shape[0]].set(obs)

    def reset(self, rng):
        state = super().reset(rng)
        obs_history = jp.zeros(15 * 31, dtype=state.obs.dtype)
        obs = self._get_obs(state.pipeline_state, state.info, obs_history)
        return state.replace(obs=obs)


def is_known_510_to_465_mismatch(error):
    """Return true only for the established 510-versus-465 shape mismatch."""
    message = str(error).lower()
    if "(510,)" not in message or "(465,)" not in message:
        return False
    return any(
        term in message
        for term in ("incompatible shapes", "incompatible", "broadcasting", "shape mismatch")
    )


def _build_policy_environment(env_class, checkpoint_path):
    policy_env = env_class(
        command_sampling="walk_trot_only", obs_noise=OBS_NOISE, kick_vel=KICK_VEL
    )
    inference_fn = load_policy(policy_env, checkpoint_path)
    eval_env = env_class(
        command_sampling="walk_trot_only", obs_noise=OBS_NOISE, kick_vel=KICK_VEL
    )
    return eval_env, inference_fn, jax.jit(eval_env.step)


def _smoke_test_policy_inference(env, inference_fn):
    reset_key, action_key = jax.random.split(jax.random.PRNGKey(9182))
    state = env.reset(reset_key)
    action, _ = inference_fn(state.obs, action_key)
    action = jax.block_until_ready(action)
    return int(state.obs.shape[0]), int(action.shape[0])


def build_compatible_policy_environment(checkpoint_path):
    """Use 510 observations first; retry only the known checkpoint mismatch."""
    try:
        env, inference_fn, jit_step = _build_policy_environment(BarkourEnv, checkpoint_path)
        observation_dimension, action_dimension = _smoke_test_policy_inference(
            env, inference_fn
        )
        if observation_dimension != CURRENT_OBSERVATION_METADATA["observation_dimension"]:
            raise RuntimeError(f"Unexpected current observation dimension: {observation_dimension}")
        return env, inference_fn, jit_step, dict(
            CURRENT_OBSERVATION_METADATA, action_dimension=action_dimension
        )
    except Exception as error:
        if not is_known_510_to_465_mismatch(error):
            raise
        print(
            "[info] checkpoint rejected the 510-observation layout; "
            "retrying the strict legacy 465-observation layout",
            flush=True,
        )

    env, inference_fn, jit_step = _build_policy_environment(
        LegacyObsBarkourEnv, checkpoint_path
    )
    observation_dimension, action_dimension = _smoke_test_policy_inference(
        env, inference_fn
    )
    if observation_dimension != LEGACY_OBSERVATION_METADATA["observation_dimension"]:
        raise RuntimeError(f"Unexpected legacy observation dimension: {observation_dimension}")
    return env, inference_fn, jit_step, dict(
        LEGACY_OBSERVATION_METADATA, action_dimension=action_dimension
    )


def get_robot_mass(env):
    """Match the reference evaluator's non-world body-mass convention."""
    body_mass = np.asarray(env.sys.mj_model.body_mass, dtype=np.float64)
    return float(body_mass.sum()) if body_mass.size <= 1 else float(body_mass[1:].sum())


def get_local_forward_speed(state):
    """Match the reference evaluator's body-0 local-forward-speed convention."""
    local_velocity = brax_math.rotate(
        state.pipeline_state.xd.vel[0],
        brax_math.quat_inv(state.pipeline_state.x.rot[0]),
    )
    return float(local_velocity[0])


def get_torso_xy_position(state, torso_idx):
    """Match the reference evaluator's torso-index-minus-one position convention."""
    return np.asarray(
        jax.device_get(state.pipeline_state.x.pos[torso_idx - 1, :2]), dtype=np.float64
    )


def get_actuated_joint_power(state):
    """Return sum(abs(tau * qd)) over actuated coordinates qd[6:]."""
    tau = np.asarray(jax.device_get(state.pipeline_state.qfrc_actuator), dtype=np.float64)
    qd = np.asarray(jax.device_get(state.pipeline_state.qd), dtype=np.float64)
    n = min(tau.size, qd.size)
    return float(np.sum(np.abs(tau[:n][6:] * qd[:n][6:])))


def evaluate_metrics(
    survived,
    post_warm_local_vx,
    vx_cmd,
    total_energy,
    robot_mass,
    planar_distance,
):
    """Apply the locked survival, success, and CoT definitions."""
    avg_speed = (
        float(sum(post_warm_local_vx) / len(post_warm_local_vx))
        if post_warm_local_vx
        else float("nan")
    )
    success = survived and 0.85 * vx_cmd <= avg_speed <= 1.15 * vx_cmd
    denominator = robot_mass * GRAVITY * planar_distance
    cot = total_energy / denominator if denominator > 0.0 else float("nan")
    return {
        "avg_speed_post_warm": avg_speed,
        "cot": float(cot),
        "survive": float(survived),
        "success": float(success),
    }


def rollout_once(env, inference_fn, jit_step, rng, command, robot_mass):
    state = reset_eval_state_with_command(env, rng, command)
    previous_xy = get_torso_xy_position(state, env._torso_idx)
    total_energy = 0.0
    planar_distance = 0.0
    post_warm_local_vx = []
    survived = True
    terminated_step = HORIZON_STEPS

    for step_index in range(HORIZON_STEPS):
        rng, action_rng = jax.random.split(rng)
        action, _ = inference_fn(state.obs, action_rng)
        state = jit_step(state, action)
        total_energy += get_actuated_joint_power(state) * float(env.dt)
        current_xy = get_torso_xy_position(state, env._torso_idx)
        planar_distance += float(np.linalg.norm(current_xy - previous_xy))
        previous_xy = current_xy
        if step_index >= WARMUP_STEPS:
            post_warm_local_vx.append(get_local_forward_speed(state))
        if float(state.done) != 0.0:
            survived = False
            terminated_step = step_index + 1
            break

    metrics = evaluate_metrics(
        survived,
        post_warm_local_vx,
        float(command[0]),
        total_energy,
        robot_mass,
        planar_distance,
    )
    return {
        "vx_cmd": float(command[0]),
        "vy_cmd": float(command[1]),
        "yaw_cmd": float(command[2]),
        "executed_steps": int(terminated_step),
        "planar_distance": planar_distance,
        "total_energy": total_energy,
        **metrics,
    }


def summarize(values):
    """Return population moments for a fixed set of 20 rollout measurements."""
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "var": variance, "std": math.sqrt(variance)}


def summarize_command(rows, vx_cmd):
    cot = summarize([row["cot"] for row in rows])
    survive = summarize([row["survive"] for row in rows])
    success = summarize([row["success"] for row in rows])
    speed = summarize([row["avg_speed_post_warm"] for row in rows])
    return {
        "vx_cmd": vx_cmd,
        "cot_mean": cot["mean"],
        "cot_var": cot["var"],
        "cot_std": cot["std"],
        "survive_mean": survive["mean"],
        "survive_var": survive["var"],
        "survive_std": survive["std"],
        "success_mean": success["mean"],
        "success_var": success["var"],
        "success_std": success["std"],
        "avg_speed_post_warm_mean": speed["mean"],
        "avg_speed_post_warm_var": speed["var"],
        "avg_speed_post_warm_std": speed["std"],
    }


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reserve_output_dir(output_dir):
    """Create a new evaluation directory; never overwrite prior results."""
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to reuse evaluation output directory: {output_dir}"
        ) from error


def evaluate_checkpoint(checkpoint_path, output_dir, velocities, seed):
    """Run the locked 20×500 protocol on one Orbax policy and write a new directory."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    output_dir = Path(output_dir).expanduser().resolve()
    reserve_output_dir(output_dir)

    env, inference_fn, jit_step, observation_metadata = build_compatible_policy_environment(
        checkpoint_path
    )
    robot_mass = get_robot_mass(env)
    rng = jax.random.PRNGKey(seed)
    rollout_rows = []
    summary_rows = []

    for vx_cmd in velocities:
        command = jp.asarray([vx_cmd, 0.0, 0.0], dtype=jp.float32)
        command_rows = []
        for rollout_index in range(NUM_TESTS):
            rng, rollout_rng = jax.random.split(rng)
            row = rollout_once(
                env, inference_fn, jit_step, rollout_rng, command, robot_mass
            )
            row["rollout_index"] = rollout_index
            row.update(observation_metadata)
            rollout_rows.append(row)
            command_rows.append(row)
        summary_rows.append(summarize_command(command_rows, vx_cmd))

    save_csv(output_dir / "rollouts.csv", rollout_rows)
    save_csv(output_dir / "summary.csv", summary_rows)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "protocol": {
            "tests_per_command": NUM_TESTS,
            "horizon_steps": HORIZON_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "velocities": [float(value) for value in velocities],
            "vy_cmd": 0.0,
            "yaw_cmd": 0.0,
            "obs_noise": OBS_NOISE,
            "kick_vel": KICK_VEL,
            "cot": "sum(abs(tau[6:] * qd[6:])) * dt / (mass * 9.81 * planar_distance)",
            "mass_convention": "sum(body_mass[1:])",
            "speed_convention": "body 0 local forward velocity",
            "position_convention": "x.pos[torso_idx - 1, :2]",
        },
        "robot_mass_kg": robot_mass,
        "observation_compatibility": observation_metadata,
        "summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rollout_rows)} rollouts to {output_dir / 'rollouts.csv'}")
    print(f"Saved command summaries to {output_dir / 'summary.csv'}")
    print(f"Saved audit summary to {output_dir / 'summary.json'}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_checkpoint(args.ckpt_path, args.output_dir, DEFAULT_VELOCITIES, args.seed)


if __name__ == "__main__":
    main()
