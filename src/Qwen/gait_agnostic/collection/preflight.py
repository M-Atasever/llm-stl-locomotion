#!/usr/bin/env python3
"""Run collector preflight without emitting collection outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gait",
        choices=("all", "walk", "trot"),
        default="all",
        help="Run the one-step inference/JIT gate for one or both expert policies.",
    )
    parser.add_argument("--walk-policy", type=Path, default=None)
    parser.add_argument("--trot-policy", type=Path, default=None)
    return parser.parse_args()


def _check_assets() -> None:
    from Barkour import BARKOUR_ROOT_PATH

    if str(BARKOUR_ROOT_PATH) == "PLACEHOLDER":
        raise FileNotFoundError(
            "PLACEHOLDER: set BARKOUR_ROOT_PATH to the directory containing scene_mjx.xml."
        )
    asset_root = BARKOUR_ROOT_PATH
    model_path = asset_root / "scene_mjx.xml"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing Barkour asset {model_path}")


def _run_gait_preflight(gait: str, checkpoint_path: Path) -> dict[str, object]:
    import jax
    from jax import numpy as jp
    import numpy as np

    from collector import (
        EXPECTED_POLICY_HASHES,
        _preflight_one_step,
        assert_expected_checkpoint_hash,
        build_compatible_policy_environment,
    )

    if str(checkpoint_path) == "PLACEHOLDER":
        raise FileNotFoundError(
            f"PLACEHOLDER: supply --{gait}-policy with the matching expert checkpoint directory."
        )
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_hash = assert_expected_checkpoint_hash(
        gait, checkpoint_path, EXPECTED_POLICY_HASHES
    )
    env, inference_fn, jit_step, observation_metadata = (
        build_compatible_policy_environment(checkpoint_path)
    )
    midpoint = 0.4 if gait == "walk" else 1.05
    command = jp.asarray([midpoint, 0.0, 0.0], dtype=jp.float32)
    _preflight_one_step(env, inference_fn, jit_step, command)
    state = env.reset(jax.random.PRNGKey(7))
    if not bool(np.all(np.isfinite(np.asarray(jax.device_get(state.obs))))):
        raise RuntimeError("Non-finite observation emitted during preflight")
    return {
        "gait": gait,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "observation_layout": observation_metadata["observation_layout"],
        "observation_dimension": observation_metadata["observation_dimension"],
    }


def main() -> None:
    args = parse_args()
    from collector import DEFAULT_TROT_POLICY, DEFAULT_WALK_POLICY

    if args.walk_policy is None:
        args.walk_policy = DEFAULT_WALK_POLICY
    if args.trot_policy is None:
        args.trot_policy = DEFAULT_TROT_POLICY
    _check_assets()
    print("Model asset: PASS")
    selected = ("walk", "trot") if args.gait == "all" else (args.gait,)
    policies = {"walk": args.walk_policy, "trot": args.trot_policy}
    results = [_run_gait_preflight(gait, policies[gait]) for gait in selected]
    print("One-step inference restore: PASS")
    print("JIT environment step: PASS")
    print("Exact score schema: PASS")
    print("Finite reward and metrics: PASS")
    print("Fixed command preflight: PASS")
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
