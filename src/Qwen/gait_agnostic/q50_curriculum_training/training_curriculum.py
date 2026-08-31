#!/usr/bin/env python3
"""Train the predeclared full-range command curriculum using the q50 reward."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import jax
from jax import numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from curriculum_config import (
    CONDITION,
    FULL_TRAINING_STAGES,
    Q50_REWARD_SOURCE,
    resolve_output_root,
    select_stage,
)


TRAINING_ROOT = Path(__file__).resolve().parent
INVOKING_WORKING_DIRECTORY = Path.cwd()
Q50_SOURCE_DIR = (TRAINING_ROOT / Q50_REWARD_SOURCE).resolve().parent
if not Q50_SOURCE_DIR.is_dir():
    raise FileNotFoundError(f"Required immutable q50 source directory is absent: {Q50_SOURCE_DIR}")
# Load only the retained reward runtime; do not change the caller's directory.
sys.path.insert(0, str(Q50_SOURCE_DIR))

from Barkour import BarkourEnv  # noqa: E402
from training import (  # noqa: E402
    FULL_BATCH_SIZE,
    FULL_NUM_ENVS,
    FULL_NUM_MINIBATCHES,
    NETWORK,
    SEED,
    _as_jsonable,
    _save_orbax_checkpoint,
    domain_randomize,
)


class CurriculumBarkourEnv(BarkourEnv):
    """Immutable q50 environment whose only change is a static vx interval."""

    def __init__(self, *, vx_min: float, vx_max: float, **kwargs):
        super().__init__(command_sampling="walk_trot_only", **kwargs)
        self._curriculum_vx_min = float(vx_min)
        self._curriculum_vx_max = float(vx_max)

    def sample_command(self, rng: jax.Array) -> jax.Array:
        base_command = super().sample_command(rng)
        vx_key, _ = jax.random.split(rng)
        vx = jax.random.uniform(
            vx_key,
            shape=(),
            minval=self._curriculum_vx_min,
            maxval=self._curriculum_vx_max,
        )
        return jp.asarray([vx, base_command[1], base_command[2]], dtype=jp.float32)


def _prepare_output_root(output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to reuse curriculum output root: {output_root}") from error


def _save_stage_policy(path: Path, params) -> None:
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params, save_args=orbax_utils.save_args_from_target(params))


def run_stage(stage, output_root: Path, restore_checkpoint_path: Path | None, smoke: bool) -> Path:
    stage_output = output_root / stage.name
    stage_output.mkdir()
    checkpoint_dir = stage_output / "checkpoints"
    checkpoint_dir.mkdir()
    metrics_path = stage_output / "metrics.jsonl"
    num_timesteps = 1_048_576 if smoke else stage.num_timesteps
    num_envs = 256 if smoke else FULL_NUM_ENVS
    num_evals = 1 if smoke else stage.num_evals
    num_minibatches = 8 if smoke else FULL_NUM_MINIBATCHES
    stage_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "stage": stage.name,
        "operational_smoke": smoke,
        "experimental": not smoke,
        "num_timesteps": num_timesteps,
        "num_evals": num_evals,
        "num_envs": num_envs,
        "batch_size": FULL_BATCH_SIZE,
        "num_minibatches": num_minibatches,
        "seed": SEED,
        "network": list(NETWORK),
        "command_sampling": {"vx_uniform": [stage.vx_min, stage.vx_max], "vy_yaw": "q50 baseline"},
        "q50_reward_source": str(Q50_SOURCE_DIR / "stl_reward.py"),
        "restore_checkpoint_path": None if restore_checkpoint_path is None else str(restore_checkpoint_path),
        "transfer_type": "from_scratch" if restore_checkpoint_path is None else "predeclared_prior_stage",
    }
    (stage_output / "run_config.json").write_text(
        json.dumps(stage_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        _save_orbax_checkpoint(checkpoint_dir / str(current_step), params)

    def progress(num_steps, metrics):
        record = {"num_steps": int(num_steps)}
        record.update({key: _as_jsonable(value) for key, value in metrics.items()})
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": stage.name, **record}, sort_keys=True), flush=True)

    env = CurriculumBarkourEnv(vx_min=stage.vx_min, vx_max=stage.vx_max)
    eval_env = CurriculumBarkourEnv(vx_min=stage.vx_min, vx_max=stage.vx_max)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=num_minibatches,
        num_updates_per_batch=4,
        discounting=0.955,
        learning_rate=0.00015,
        entropy_cost=0.004,
        num_envs=num_envs,
        batch_size=FULL_BATCH_SIZE,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=NETWORK
        ),
        randomization_fn=domain_randomize,
        policy_params_fn=policy_params_fn,
        seed=SEED,
    )
    _, params, final_metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        progress_fn=progress,
        restore_checkpoint_path=None
        if restore_checkpoint_path is None
        else str(restore_checkpoint_path.resolve()),
    )
    final_policy = stage_output / "final_policy"
    _save_stage_policy(final_policy, params)
    (stage_output / "final_metrics.json").write_text(
        json.dumps({key: _as_jsonable(value) for key, value in final_metrics.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return final_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=[stage.name for stage in FULL_TRAINING_STAGES])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_stages = (
        (select_stage(args.stage),) if args.stage is not None else FULL_TRAINING_STAGES
    )
    output_root = resolve_output_root(args.output_root, INVOKING_WORKING_DIRECTORY).resolve()
    _prepare_output_root(output_root)
    prior_final_policy = None
    for stage in selected_stages:
        if stage.restore_from_stage is not None and prior_final_policy is None:
            raise RuntimeError("A later curriculum stage cannot run without its prior stage")
        prior_final_policy = run_stage(stage, output_root, prior_final_policy, args.smoke)
    (output_root / "curriculum_manifest.json").write_text(
        json.dumps(
            {
                "condition": CONDITION,
                "operational_smoke": args.smoke,
                "stage_names": [stage.name for stage in selected_stages],
                "total_requested_timesteps": sum(
                    1_048_576 if args.smoke else stage.num_timesteps for stage in selected_stages
                ),
                "final_policy": str(prior_final_policy),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
