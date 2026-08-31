#!/usr/bin/env python3
"""Collect trajectory data with STL robustness scores for expert policies.

This script evaluates trained expert policies and computes ALL STL robustness
scores at every timestep for both walking and trotting gaits, saving as CSV.
"""

from __future__ import annotations

import argparse
import functools
import csv
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'multi_gait_training' / 'src')
)
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')

import jax
from jax import numpy as jp
import numpy as np
from brax import math
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from stl_reward import reward_step
from coeff_config import (
    validate_history_config,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    WALK_TO_TROT_ENTER,
    H,
    H_by_mode,
)

# Add the LegacyObsBarkourEnv class for compatibility with older checkpoints
class LegacyObsBarkourEnv(BarkourEnv):
    """Barkour env with the pre-mode-flag observation layout.

    Old checkpoints may expect 31 features per frame instead of 34 because the
    3-d mode one-hot vector was not part of the observation when they were
    trained.
    """

    def _get_obs(
        self,
        pipeline_state,
        state_info: Dict[str, Any],
        obs_history,
    ):
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),
            state_info["command"] * jp.array([2.0, 2.0, 0.25]),
            pipeline_state.q[7:] - self._default_pose,
            state_info["last_act"],
        ])

        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        obs = jp.roll(obs_history, obs.shape[0]).at[:obs.shape[0]].set(obs)
        return obs

    def reset(self, rng):
        state = super().reset(rng)
        obs_history = jp.zeros(15 * 31, dtype=state.obs.dtype)
        obs = self._get_obs(state.pipeline_state, state.info, obs_history)
        return state.replace(obs=obs)


def _build_env_pair(env_cls, ckpt_path, obs_noise=0.0, kick_vel=0.0):
    env_for_policy = env_cls(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_inference_fn = load_policy(env_for_policy, ckpt_path)
    eval_env = env_cls(obs_noise=obs_noise, kick_vel=kick_vel)
    jit_step = jax.jit(eval_env.step)
    return env_for_policy, eval_env, jit_inference_fn, jit_step


def _smoke_test_policy_inference(env, inference_fn, seed: int = 123):
    """Force one policy call so checkpoint/observation-layout mismatches surface early."""
    rng = jax.random.PRNGKey(seed)
    reset_rng, act_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    ctrl, _ = inference_fn(state.obs, act_rng)
    ctrl = jax.block_until_ready(ctrl)
    return int(state.obs.shape[0]), int(ctrl.shape[0])


def _is_obs_shape_mismatch_error(err: BaseException) -> bool:
    msg = str(err)
    return (
        "incompatible shapes for broadcasting" in msg
        or ("broadcasting" in msg and "sub got incompatible shapes" in msg)
        or ("(510,)" in msg and "(465,)" in msg)
        or ("(465,)" in msg and "(510,)" in msg)
    )


def build_compatible_envs(ckpt_path, obs_noise=0.0, kick_vel=0.0):
    # First try the current observation layout: 15 frames * 34 features = 510.
    try:
        _, eval_env, jit_inference_fn, jit_step = _build_env_pair(
            BarkourEnv, ckpt_path, obs_noise, kick_vel
        )
        obs_dim, action_dim = _smoke_test_policy_inference(eval_env, jit_inference_fn)
        print(
            f"[info] using current observation layout: obs_dim={obs_dim}, "
            f"action_dim={action_dim}"
        )
        obs_layout = "with_mode_flag"
        return eval_env, jit_inference_fn, jit_step, obs_layout
    except Exception as e:
        if not _is_obs_shape_mismatch_error(e):
            raise
        print(
            "[info] current 510-dim observation layout does not match this checkpoint; "
            "retrying legacy 465-dim observation layout..."
        )

    # Fall back to the old observation layout: 15 frames * 31 features = 465.
    _, eval_env, jit_inference_fn, jit_step = _build_env_pair(
        LegacyObsBarkourEnv, ckpt_path, obs_noise, kick_vel
    )
    obs_dim, action_dim = _smoke_test_policy_inference(eval_env, jit_inference_fn)
    print(
        f"[info] using legacy observation layout: obs_dim={obs_dim}, "
        f"action_dim={action_dim}"
    )
    obs_layout = "legacy_no_mode_flag"
    return eval_env, jit_inference_fn, jit_step, obs_layout


xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def _install_running_stats_compat_patch():
    try:
        from brax.training.acme import running_statistics
    except Exception:
        return lambda: None

    cls = running_statistics.RunningStatisticsState
    dataclass_fields = getattr(cls, '__dataclass_fields__', {})
    if 'std_eps' in dataclass_fields:
        return lambda: None

    orig_init = cls.__init__

    def patched_init(self, *args, std_eps=None, **kwargs):
        return orig_init(self, *args, **kwargs)

    cls.__init__ = patched_init

    def restore():
        cls.__init__ = orig_init

    return restore


def load_policy(env: BarkourEnv, ckpt_path: str):
    """Loads a PPO policy exactly in the style of the old testing script."""
    restore_patch = _install_running_stats_compat_patch()
    try:
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=0,
            episode_length=1000,
            normalize_observations=True,
            restore_checkpoint_path=ckpt_path,
            network_factory=functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(128, 128, 128, 128),
            ),
        )
    finally:
        restore_patch()

    inference_fn = make_inference_fn(params)
    return jax.jit(inference_fn)


def _mode_from_command(command: jp.ndarray) -> jp.ndarray:
    vx_abs = jp.abs(command[0])
    return jp.where(
        vx_abs >= TROT_TO_BOUND_ENTER,
        jp.array(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx_abs >= WALK_TO_TROT_ENTER,
            jp.array(MODE_TROT, dtype=jp.int32),
            jp.array(MODE_WALK, dtype=jp.int32),
        ),
    )


def reset_eval_state_with_command(env: BarkourEnv, rng: jax.Array, command: jp.ndarray):
    """Resets the environment and rebuilds observation with the desired command."""
    state = env.reset(rng)

    info = dict(state.info)
    info['command'] = command
    info['mode'] = _mode_from_command(command)
    info['history_len'] = jp.array(0, dtype=jp.int32)

    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)

    return state.replace(obs=obs, info=info)


def get_robustness_column_names() -> List[str]:
    """Get the column names for all robustness scores returned by reward_step."""
    return [
        'total_stl_reward',
        'tau_effort',
        'rho_safety',
        'rho_torque',
        'rho_comz',
        'rho_roll',
        'rho_pitch',
        'rho_slip',
        'rho_bound',
        'rho_trot',
        'rho_walk',
        'rho_v_x',
        'rho_v_y',
        'rho_yaw',
        'rho_nlegs',
        'rho_gait',
        'rho_v_x_error',
        'rho_v_y_error',
        'rho_yaw_error',
        'rho_diag2',
        'rho_stride',
        'rho_duty',
        'rho_3plus_event',
        'rho_support',
        'rho_diag_phase',
        'rho_p2',
        'rho_tracking',
        'rho_timing',
        'rho_pattern',
        'rho_front',
        'rho_hind',
        'rho_hindfront',
        'rho_flight',
        'rho_front_only',
        'rho_hind_only',
        'rho_all4',
        'rho_bound_event',
        'pitch',
        'roll',
        'FL',
        'HL',
        'FR',
        'HR',
    ]


def rollout_trajectory(
    env: BarkourEnv,
    jit_inference_fn,
    jit_step,
    rng: jax.Array,
    command: jp.ndarray,
    horizon: int,
    trajectory_id: int,
) -> List[Dict[str, Any]]:
    """Rollout a single trajectory and collect ALL STL robustness scores."""
    state = reset_eval_state_with_command(env, rng, command)

    timestep_data = []

    for t in range(horizon):
        rng, act_rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)

        # Create base row data
        row_data = {
            'trajectory_id': trajectory_id,
            'timestep': t,
            'mode': int(state.info['mode']),
            'command_vx': float(state.info['command'][0]),
            'command_vy': float(state.info['command'][1]),
            'command_yaw': float(state.info['command'][2]),
            'survived': 1.0 if float(state.done) == 0.0 else 0.0,
            'done': float(state.done),
            'reward': float(state.reward),
        }

        # Compute STL robustness after sufficient history is available.
        if state.info['history_len'] >= 8:  # Minimum warmup length
            reward_input = {
                'tau_history': state.info['tau_history'],
                'CoM_history': state.info['CoM_history'],
                'contact_history': state.info['contact_history'],
                'feet_history': state.info['feet_history'],
                'lin_velocity_history': state.info['lin_velocity_history'],
                'ang_velocity_history': state.info['ang_velocity_history'],
                'com_z_history': state.info['com_z_history'],
                'roll_history': state.info['roll_history'],
                'pitch_history': state.info['pitch_history'],
                'slipmax_history': state.info['slipmax_history'],
            }

            robustness_scores = compute_stl_robustness(
                reward_input,
                state.info['command'],
                state.info['mode'],
                state.info['history_len']
            )

            # Add all robustness scores to row data
            column_names = get_robustness_column_names()
            for i, col_name in enumerate(column_names):
                row_data[col_name] = float(robustness_scores[i])
        else:
            # For initial steps without enough history, set robustness to NaN
            column_names = get_robustness_column_names()
            for col_name in column_names:
                row_data[col_name] = float('nan')

        timestep_data.append(row_data)

        # Stop if terminated
        if float(state.done) != 0.0:
            break

    return timestep_data


def compute_stl_robustness(reward_input, command, mode, valid_len):
    """Compute STL robustness scores using the existing reward_step function."""
    return reward_step(
        reward_input=reward_input,
        commands=command,
        mode=mode,
        valid_len=valid_len,
        weights_override=None,
    )


def save_timesteps_to_csv(timesteps: List[Dict[str, Any]], output_file: Path):
    """Save all timestep data to a single CSV file."""
    if not timesteps:
        return

    # Get all column names from the first row
    all_columns = list(timesteps[0].keys())

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(timesteps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--walk-ckpt', type=str, required=True, help='Walk expert checkpoint path')
    parser.add_argument('--trot-ckpt', type=str, required=True, help='Trot expert checkpoint path')
    parser.add_argument('--num-trajectories', type=int, default=50)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='stl_trajectory_data')
    parser.add_argument('--obs-noise', type=float, default=0.0)
    parser.add_argument('--kick-vel', type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_history_config()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk evaluation
    print("Evaluating walk expert policy...")
    walk_env, walk_inference_fn, walk_jit_step, walk_obs_layout = build_compatible_envs(
        args.walk_ckpt, args.obs_noise, args.kick_vel
    )

    walk_all_timesteps = []
    walk_rng = jax.random.PRNGKey(args.seed)

    # Use walking command (below WALK_TO_TROT_ENTER = 0.72)
    walk_command = jp.array([0.5, 0.0, 0.0], dtype=jp.float32)

    for i in range(args.num_trajectories):
        print(f"  Walk trajectory {i+1}/{args.num_trajectories}")
        walk_rng, traj_rng = jax.random.split(walk_rng)
        traj_data = rollout_trajectory(
            walk_env, walk_inference_fn, walk_jit_step,
            traj_rng, walk_command, args.horizon, i
        )
        walk_all_timesteps.extend(traj_data)

    # Save walk data
    walk_output_file = output_dir / 'walk_trajectories.csv'
    save_timesteps_to_csv(walk_all_timesteps, walk_output_file)
    print(f"Saved walk trajectories to {walk_output_file}")

    # Trot evaluation
    print("Evaluating trot expert policy...")
    trot_env, trot_inference_fn, trot_jit_step, trot_obs_layout = build_compatible_envs(
        args.trot_ckpt, args.obs_noise, args.kick_vel
    )

    trot_all_timesteps = []
    trot_rng = jax.random.PRNGKey(args.seed + 1000)

    # Use trotting command (between WALK_TO_TROT_ENTER = 0.72 and TROT_TO_BOUND_ENTER = 1.55)
    trot_command = jp.array([1.2, 0.0, 0.0], dtype=jp.float32)

    for i in range(args.num_trajectories):
        print(f"  Trot trajectory {i+1}/{args.num_trajectories}")
        trot_rng, traj_rng = jax.random.split(trot_rng)
        traj_data = rollout_trajectory(
            trot_env, trot_inference_fn, trot_jit_step,
            traj_rng, trot_command, args.horizon, i
        )
        trot_all_timesteps.extend(traj_data)

    # Save trot data
    trot_output_file = output_dir / 'trot_trajectories.csv'
    save_timesteps_to_csv(trot_all_timesteps, trot_output_file)
    print(f"Saved trot trajectories to {trot_output_file}")

    print("\nData collection completed!")
    print(f"Walk data: {walk_output_file} ({len(walk_all_timesteps)} timesteps)")
    print(f"Trot data: {trot_output_file} ({len(trot_all_timesteps)} timesteps)")


if __name__ == '__main__':
    main()
