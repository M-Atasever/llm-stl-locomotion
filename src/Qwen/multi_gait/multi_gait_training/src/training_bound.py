#!/usr/bin/env python3
"""Continue the multi-gait replicate for an additional 400M with bound enabled."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
    xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from Barkour import BarkourEnv
from coeff_config import validate_history_config


FULL_NUM_TIMESTEPS = 400_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restore-checkpoint", type=Path, required=True)
    parser.add_argument("--num-timesteps", type=int, default=FULL_NUM_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short operational PPO smoke test, not an experiment.",
    )
    return parser.parse_args()


def _as_jsonable(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def domain_randomize(sys, rng):
    """Apply the reference implementation's domain randomization."""

    @jax.vmap
    def rand(key):
        _, key = jax.random.split(key, 2)
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)

        _, key = jax.random.split(key, 2)
        parameter = jax.random.uniform(key, (1,), minval=-5, maxval=5)
        parameter = parameter + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(parameter)
        bias = sys.actuator_biasprm.at[:, 1].set(-parameter)
        return friction, gain, bias

    friction, gain, bias = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )
    randomized = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )
    return randomized, in_axes


def main() -> None:
    args = parse_args()
    validate_history_config()
    if args.smoke:
        args.num_timesteps = 1_048_576
        args.num_envs = 256
        args.num_evals = 1
        args.batch_size = 256
        args.num_minibatches = 8

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    final_policy_dir = output_dir / "final_policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operational_smoke": args.smoke,
        "num_timesteps": args.num_timesteps,
        "num_evals": args.num_evals,
        "reward_scaling": 1,
        "episode_length": 1000,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 30,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": 4,
        "discounting": 0.955,
        "learning_rate": 0.00015,
        "entropy_cost": 0.004,
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "command_sampling": "mixed",
        "bound_training": True,
        "continuation_requested_timesteps": args.num_timesteps,
        "restore_checkpoint": str(args.restore_checkpoint.expanduser().resolve()),
        "replicate": "multi_gait_qwen3.6-plus",
        "condition_id": "multi_gait_component_q50_baseline_weight",
        "filter_protocol": "component_q50",
        "tracking_weights_by_mode": [1.0, 1.0, 1.15],
        "walking_effective_gait_score": "smooth_min(rho_no_flight, rho_duty_factor_high)",
        "walking_excluded_gait_components": ["rho_three_plus_contact"],
        "trotting_effective_gait_score": "smooth_min(rho_two_contact_dominance, rho_diagonal_sync_error, rho_low_flight_time)",
        "trotting_excluded_gait_components": ["rho_diagonal_pair_support"],
        "protected_velocity_tracking_scores": ["rho_tracking", "Vel_track_x"],
        "diagnostic_score_schema_retained": True,
        "weights_or_thresholds_modified": True,
        "parameter_updates": {
            "p_3plus_min_walk": "K_REQUIRE_3PLUS / H_by_mode[walk] = 11/30",
            "base_height": "one-sided Base_height >= 0.16; upper bound removed"
        },
        "velocity_tracking_definition": "abs(vx_actual - vx_command)",
        "vy_yaw_tracking_active": False,
        "nominal_checkpoint_milestones": [100000000, 200000000, 300000000, 400000000],
        "velocity_sampling_modified": False,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = epath.Path(checkpoint_dir / str(current_step))
        checkpointer.save(path, params, force=True, save_args=save_args)

    def progress(num_steps, metrics):
        record = {"num_steps": int(num_steps)}
        record.update({key: _as_jsonable(value) for key, value in metrics.items()})
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "num_steps": record["num_steps"],
                    "eval/episode_reward": record.get("eval/episode_reward"),
                    "eval/episode_total_stl_reward": record.get(
                        "eval/episode_total_stl_reward"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=4,
        discounting=0.955,
        learning_rate=0.00015,
        entropy_cost=0.004,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=str(args.restore_checkpoint.expanduser().resolve()),
        seed=args.seed,
    )

    env = BarkourEnv(command_sampling="mixed", bound_straight_commands=True)
    eval_env = BarkourEnv(command_sampling="mixed", bound_straight_commands=True)
    make_inference_fn, params, final_metrics = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )
    del make_inference_fn

    model.save_params(str(final_policy_dir), params)
    final_metrics_json = {
        key: _as_jsonable(value) for key, value in final_metrics.items()
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Final policy: {final_policy_dir}")


if __name__ == "__main__":
    main()
