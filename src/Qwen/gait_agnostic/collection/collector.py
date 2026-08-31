#!/usr/bin/env python3
"""Collect per-timestep STL robustness from the walk and trot expert policies.

The two gaits can be collected together or independently.  Each selected gait
produces exactly 50 trajectories of 500 simulator steps, one raw CSV, one raw
compressed NPZ, and a metadata JSON file.  This collector deliberately performs
no q50 calculation, filtering, or reward-component selection.  Any expert
termination is treated as a failed collection and stops immediately.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Mapping

def choose_mujoco_gl_backend(
    platform_name: str | None = None, configured_value: str | None = None
) -> str:
    if platform_name is None:
        platform_name = sys.platform
    if platform_name == "darwin":
        return "glfw"
    return configured_value or "egl"


def configure_mujoco_gl_backend() -> str:
    backend = choose_mujoco_gl_backend(sys.platform, os.environ.get("MUJOCO_GL"))
    os.environ["MUJOCO_GL"] = backend
    return backend


configure_mujoco_gl_backend()
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_triton_gemm_any=True"
    ).strip()

TRAINING_ROOT = Path(__file__).resolve().parent
RUN_ROOT = TRAINING_ROOT.parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import jax
from jax import numpy as jp
import numpy as np
from brax import math
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from coeff_config import (
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    WALK_TO_TROT_ENTER,
)
from stl_reward import METRIC_KEYS


N_TRAJECTORIES = 50
N_STEPS = 500
DEFAULT_SEED = 0
OBS_NOISE = 0.0
KICK_VEL = 0.0
RUN_ID = "gait_agnostic"
# The archive did not include the referenced raw collector or request manifest.
RAW_GENERATED_COLLECTOR = "PLACEHOLDER"
PROMPT3_MANIFEST = "PLACEHOLDER"
RAW_GENERATED_COLLECTOR_SHA256 = "PLACEHOLDER"
EXPECTED_POLICY_HASHES = {
    "walk": "86692b0cdaa5d3fc2f63a5c8fedcfdc9db23460b0e664ff7042f1d6d74617f4f",
    "trot": "c72f6b56530365f15eccbbe0b76443ca097c426ed16c1d563369d1e31f2caca7",
}
ENVIRONMENT_LOCK_INFO = {
    "obs_noise": OBS_NOISE,
    "kick_vel": KICK_VEL,
    "fixed_command_after_reset": True,
    "fixed_command_reasserted_after_every_step": True,
}

MODE_NAMES = {
    MODE_WALK: "walk",
    MODE_TROT: "trot",
    MODE_BOUND: "bound",
}

INDIVIDUAL_SCORE_KEYS = (
    "vel_track_x",
    "safe_height",
    "safe_orientation",
    "low_slip",
)
GROUPED_SCORE_KEYS = (
    "rho_task_safety",
    "rho_gait_quality",
    "total_stl_reward",
)
DIAGNOSTIC_KEYS = (
    "diagnostic_abs_velocity_error_y",
    "diagnostic_abs_velocity_error_yaw",
)
SCORE_KEYS = INDIVIDUAL_SCORE_KEYS + GROUPED_SCORE_KEYS
ALL_RECORDED_METRIC_KEYS = SCORE_KEYS + DIAGNOSTIC_KEYS

DEFAULT_WALK_POLICY = Path("PLACEHOLDER")
DEFAULT_TROT_POLICY = Path("PLACEHOLDER")

GAIT_PROTOCOLS = {
    "walk": {
        "vx_min": 0.3,
        "vx_max": 0.5,
        "expected_mode": MODE_WALK,
    },
    "trot": {
        "vx_min": 0.9,
        "vx_max": 1.2,
        "expected_mode": MODE_TROT,
    },
}

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


class ExpertTerminationError(RuntimeError):
    """The protocol forbids continuing after any expert termination."""


def _install_running_stats_compat_patch() -> Callable[[], None]:
    """Accept checkpoints serialized by Brax versions that included std_eps."""
    try:
        from brax.training.acme import running_statistics
    except Exception:
        return lambda: None

    cls = running_statistics.RunningStatisticsState
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    if "std_eps" in dataclass_fields:
        return lambda: None

    original_init = cls.__init__

    def patched_init(self, *args, std_eps=None, **kwargs):
        del std_eps
        return original_init(self, *args, **kwargs)

    cls.__init__ = patched_init

    def restore() -> None:
        cls.__init__ = original_init

    return restore


def load_policy(env: BarkourEnv, checkpoint_path: Path):
    """Restore the 4x128 PPO expert and return its jitted inference function."""
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


def _mode_from_command(command: jax.Array) -> jax.Array:
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


def reset_eval_state_with_command(
    env: BarkourEnv, rng: jax.Array, command: jax.Array
):
    """Reset and rebuild the first observation with the requested fixed command."""
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = _mode_from_command(command)
    info["history_len"] = jp.asarray(0, dtype=jp.int32)
    obs_history = jp.zeros_like(state.obs)
    obs = env._get_obs(state.pipeline_state, info, obs_history)
    return state.replace(obs=obs, info=info)


def preserve_fixed_command(
    env: BarkourEnv,
    state,
    command: jax.Array,
    previous_observation: jax.Array,
):
    """Undo any command resampling in ``step`` and rebuild the current frame.

    BarkourEnv can sample a replacement command when a termination is detected.
    Collection fails immediately on any expert termination, but every successful
    step before that failure must still preserve the trajectory's original
    fixed command.
    """
    info = dict(state.info)
    info["command"] = command
    info["mode"] = _mode_from_command(command)
    obs = env._get_obs(state.pipeline_state, info, previous_observation)
    return state.replace(obs=obs, info=info)


# Compatibility implementation supplied by the gait-agnostic pipeline.  Expert
# checkpoints may normalize 465 observations (15 frames x 31 features), while
# the current environment has 510 (15 x 34) after adding mode_one_hot.
class LegacyObsBarkourEnv(BarkourEnv):
    """Barkour environment with the pre-mode-one-hot observation layout."""

    def _get_obs(
        self,
        pipeline_state,
        state_info: dict[str, Any],
        obs_history,
    ):
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        obs = jp.concatenate(
            [
                jp.asarray([local_rpyrate[2]]) * 0.25,
                math.rotate(jp.asarray([0, 0, -1]), inv_torso_rot),
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


def is_known_510_to_465_mismatch(error: BaseException) -> bool:
    """Recognize only the known current-vs-legacy observation mismatch."""
    message = str(error)
    has_510 = "(510,)" in message
    has_465 = "(465,)" in message
    if not (has_510 and has_465):
        return False
    mismatch_terms = (
        "incompatible shapes",
        "incompatible",
        "broadcasting",
        "shape mismatch",
    )
    return any(term in message.lower() for term in mismatch_terms)


def _build_policy_environment(
    env_class: type[BarkourEnv], checkpoint_path: Path
):
    policy_env = env_class(obs_noise=OBS_NOISE, kick_vel=KICK_VEL)
    inference_fn = load_policy(policy_env, checkpoint_path)
    eval_env = env_class(obs_noise=OBS_NOISE, kick_vel=KICK_VEL)
    return eval_env, inference_fn, jax.jit(eval_env.step)


def _smoke_test_policy_inference(
    env: BarkourEnv, inference_fn, seed: int = 9182
) -> tuple[int, int]:
    reset_key, action_key = jax.random.split(jax.random.PRNGKey(seed))
    state = env.reset(reset_key)
    action, _ = inference_fn(state.obs, action_key)
    action = jax.block_until_ready(action)
    return int(state.obs.shape[0]), int(action.shape[0])


def build_compatible_policy_environment(checkpoint_path: Path):
    """Try 510 observations, then fall back only on the known 465 mismatch."""
    try:
        env, inference_fn, jit_step = _build_policy_environment(
            BarkourEnv, checkpoint_path
        )
        obs_dim, action_dim = _smoke_test_policy_inference(env, inference_fn)
        if obs_dim != CURRENT_OBSERVATION_METADATA["observation_dimension"]:
            raise RuntimeError(f"Unexpected current observation dimension: {obs_dim}")
        return (
            env,
            inference_fn,
            jit_step,
            dict(CURRENT_OBSERVATION_METADATA, action_dimension=action_dim),
        )
    except Exception as error:
        if not is_known_510_to_465_mismatch(error):
            raise
        print(
            "[info] checkpoint rejected the 510-observation layout; "
            "retrying the supplied legacy 465-observation layout",
            flush=True,
        )

    env, inference_fn, jit_step = _build_policy_environment(
        LegacyObsBarkourEnv, checkpoint_path
    )
    obs_dim, action_dim = _smoke_test_policy_inference(env, inference_fn)
    if obs_dim != LEGACY_OBSERVATION_METADATA["observation_dimension"]:
        raise RuntimeError(f"Unexpected legacy observation dimension: {obs_dim}")
    return (
        env,
        inference_fn,
        jit_step,
        dict(LEGACY_OBSERVATION_METADATA, action_dimension=action_dim),
    )


def _as_scalar(value):
    return np.asarray(jax.device_get(value)).item()


def checkpoint_tree_sha256(path: Path) -> str:
    """Hash relative names, lengths, and bytes for the complete checkpoint tree."""
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with file_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def assert_expected_checkpoint_hash(
    gait: str, checkpoint_path: Path, expected_hashes: Mapping[str, str]
) -> str:
    if str(checkpoint_path) == "PLACEHOLDER":
        raise FileNotFoundError(
            f"PLACEHOLDER: supply --{gait}-policy with the matching expert checkpoint directory."
        )
    actual_hash = checkpoint_tree_sha256(checkpoint_path)
    expected_hash = expected_hashes[gait]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Unexpected {gait} checkpoint hash: expected {expected_hash}, "
            f"got {actual_hash} for {checkpoint_path}"
        )
    return actual_hash


def reserve_output_dirs(output_root: Path, gaits: tuple[str, ...]) -> dict[str, Path]:
    """Reserve every requested gait directory before loading checkpoints."""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reserved = {gait: output_root / gait for gait in gaits}
    conflicts = [path for path in reserved.values() if path.exists()]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Selected gait output directories already exist: {joined}"
        )
    for path in reserved.values():
        path.mkdir(parents=False, exist_ok=False)
    return reserved


def enforce_no_termination(
    done_this_step: bool, gait: str, trajectory: int, timestep: int
) -> None:
    if done_this_step:
        raise ExpertTerminationError(
            f"{gait} expert terminated at trajectory {trajectory}, timestep {timestep}"
        )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = path.parent / f"{path.stem}.partial{path.suffix}"
    np.savez_compressed(temporary, **payload)
    return temporary


def _verify_score_schema(rewards: Mapping[str, Any]) -> None:
    missing = [key for key in ALL_RECORDED_METRIC_KEYS if key not in rewards]
    extra_generated = [key for key in METRIC_KEYS if key not in ALL_RECORDED_METRIC_KEYS]
    if missing:
        raise KeyError(f"Missing expected per-timestep metrics: {missing}")
    if extra_generated:
        raise RuntimeError(
            "Collector schema omitted generated STL metrics: "
            + ", ".join(extra_generated)
        )


def validate_recorded_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected_shape = (N_TRAJECTORIES, N_STEPS)
    required = {
        "trajectory",
        "timestep",
        "command_vx",
        "command_vy",
        "command_yaw",
        "mode",
        "done_this_step",
        "trajectory_terminated",
        "first_termination_timestep",
        *ALL_RECORDED_METRIC_KEYS,
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Missing arrays: {missing}")
    for key in required:
        if np.asarray(arrays[key]).shape != expected_shape:
            raise ValueError(
                f"Array {key} has shape {np.asarray(arrays[key]).shape}, "
                f"expected {expected_shape}"
            )
    finite_keys = (
        "command_vx",
        "command_vy",
        "command_yaw",
        *ALL_RECORDED_METRIC_KEYS,
    )
    for key in finite_keys:
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ValueError(f"Non-finite values detected in {key}")


def validate_csv_and_npz_match(
    csv_path: Path, npz_path: Path, metadata_path: Path
) -> dict[str, Any]:
    arrays = {key: value for key, value in np.load(npz_path, allow_pickle=False).items()}
    validate_recorded_arrays(arrays)
    csv_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_rows, row in enumerate(reader, start=1):
            trajectory = int(row["trajectory"])
            timestep = int(row["timestep"])
            for key in (
                "trajectory",
                "timestep",
                "mode",
                "first_termination_timestep",
            ):
                if int(arrays[key][trajectory, timestep]) != int(row[key]):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
            for key in ("command_vx", "command_vy", "command_yaw", *ALL_RECORDED_METRIC_KEYS):
                if not np.isclose(
                    float(arrays[key][trajectory, timestep]),
                    float(row[key]),
                    atol=1e-7,
                    rtol=0.0,
                ):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
            for key in ("done_this_step", "trajectory_terminated"):
                if int(bool(arrays[key][trajectory, timestep])) != int(row[key]):
                    raise ValueError(f"CSV/NPZ equivalence mismatch for {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_rows = N_TRAJECTORIES * N_STEPS
    if csv_rows != expected_rows:
        raise ValueError(f"CSV row count {csv_rows} does not equal {expected_rows}")
    if int(metadata.get("completed_rows", expected_rows)) != expected_rows:
        raise ValueError("Metadata completed_rows does not equal 25000")
    return {
        "expected_rows": expected_rows,
        "csv_rows": csv_rows,
        "shape": [N_TRAJECTORIES, N_STEPS],
    }


def _empty_arrays() -> dict[str, np.ndarray]:
    shape = (N_TRAJECTORIES, N_STEPS)
    arrays: dict[str, np.ndarray] = {
        "trajectory": np.zeros(shape, dtype=np.int32),
        "timestep": np.zeros(shape, dtype=np.int32),
        "command_vx": np.zeros(shape, dtype=np.float32),
        "command_vy": np.zeros(shape, dtype=np.float32),
        "command_yaw": np.zeros(shape, dtype=np.float32),
        "mode": np.zeros(shape, dtype=np.int32),
        "done_this_step": np.zeros(shape, dtype=np.bool_),
        "trajectory_terminated": np.zeros(shape, dtype=np.bool_),
        "first_termination_timestep": np.full(shape, -1, dtype=np.int32),
    }
    for key in ALL_RECORDED_METRIC_KEYS:
        arrays[key] = np.zeros(shape, dtype=np.float32)
    return arrays


def _preflight_one_step(
    env: BarkourEnv,
    inference_fn,
    jit_step,
    command: jax.Array,
) -> None:
    reset_key, action_key = jax.random.split(jax.random.PRNGKey(813))
    state = reset_eval_state_with_command(env, reset_key, command)
    previous_observation = state.obs
    action, _ = inference_fn(state.obs, action_key)
    state = jit_step(state, action)
    state = preserve_fixed_command(env, state, command, previous_observation)
    jax.block_until_ready(state.reward)
    _verify_score_schema(state.info["rewards"])
    if not np.array_equal(
        np.asarray(jax.device_get(state.info["command"])),
        np.asarray(jax.device_get(command)),
    ):
        raise RuntimeError("Fixed command preservation preflight failed")


def collect_gait(
    gait: str,
    checkpoint_path: Path,
    gait_output: Path | None,
    seed: int,
    preflight_only: bool,
) -> dict[str, Any]:
    protocol = GAIT_PROTOCOLS[gait]
    if str(checkpoint_path) == "PLACEHOLDER":
        raise FileNotFoundError(
            f"PLACEHOLDER: supply --{gait}-policy with the matching expert checkpoint directory."
        )
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"{gait} expert checkpoint directory does not exist: {checkpoint_path}"
        )

    checkpoint_hash = assert_expected_checkpoint_hash(
        gait, checkpoint_path, EXPECTED_POLICY_HASHES
    )
    env, inference_fn, jit_step, observation_metadata = (
        build_compatible_policy_environment(checkpoint_path)
    )
    midpoint_command = jp.asarray(
        [(protocol["vx_min"] + protocol["vx_max"]) / 2.0, 0.0, 0.0],
        dtype=jp.float32,
    )
    _preflight_one_step(env, inference_fn, jit_step, midpoint_command)

    base_metadata: dict[str, Any] = {
        "gait": gait,
        "status": "preflight_passed",
        "source_policy_path": str(checkpoint_path),
        "source_policy_tree_sha256": checkpoint_hash,
        "run_id": RUN_ID,
        "raw_generated_collector": str(RAW_GENERATED_COLLECTOR),
        "raw_generated_collector_sha256": RAW_GENERATED_COLLECTOR_SHA256,
        "prompt3_request_manifest": str(PROMPT3_MANIFEST),
        **observation_metadata,
        **ENVIRONMENT_LOCK_INFO,
        "trajectory_count": N_TRAJECTORIES,
        "timesteps_per_trajectory": N_STEPS,
        "expected_rows": N_TRAJECTORIES * N_STEPS,
        "command_sampling": "one uniform vx draw per trajectory",
        "command_vx_interval": [protocol["vx_min"], protocol["vx_max"]],
        "command_vy": 0.0,
        "command_yaw": 0.0,
        "fixed_command_after_reset": True,
        "fixed_command_reasserted_after_every_step": True,
        "individual_score_keys": list(INDIVIDUAL_SCORE_KEYS),
        "grouped_score_keys": list(GROUPED_SCORE_KEYS),
        "diagnostic_metric_keys": list(DIAGNOSTIC_KEYS),
        "q50_calculated": False,
        "filtering_applied": False,
    }
    if preflight_only:
        print(json.dumps(base_metadata, indent=2), flush=True)
        return base_metadata

    if gait_output is None:
        raise ValueError("gait_output must be reserved before collection")
    csv_partial = gait_output / "robustness_raw.partial.csv"
    csv_final = gait_output / "robustness_raw.csv"
    npz_partial = gait_output / "robustness_raw.partial.npz"
    npz_final = gait_output / "robustness_raw.npz"
    metadata_path = gait_output / "metadata.json"

    metadata = dict(base_metadata, status="collecting", seed=seed)
    _write_json_atomic(metadata_path, metadata)

    arrays = _empty_arrays()
    velocity_rng = np.random.default_rng(seed)
    sampled_vx = velocity_rng.uniform(
        protocol["vx_min"], protocol["vx_max"], size=N_TRAJECTORIES
    ).astype(np.float32)

    fieldnames = [
        "gait",
        "trajectory",
        "timestep",
        "command_vx",
        "command_vy",
        "command_yaw",
        "mode",
        "mode_name",
        "done_this_step",
        "trajectory_terminated",
        "first_termination_timestep",
        "source_policy_path",
        "source_policy_tree_sha256",
        "observation_layout",
        "observation_dimension",
        "observation_history_frames",
        "observation_features_per_frame",
        "observation_mode_one_hot_included",
        *ALL_RECORDED_METRIC_KEYS,
    ]

    emitted_rows = 0
    try:
        with csv_partial.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for trajectory in range(N_TRAJECTORIES):
                command = jp.asarray(
                    [sampled_vx[trajectory], 0.0, 0.0], dtype=jp.float32
                )
                expected_mode = int(protocol["expected_mode"])
                command_mode = int(_as_scalar(_mode_from_command(command)))
                if command_mode != expected_mode:
                    raise RuntimeError(
                        f"{gait} vx={sampled_vx[trajectory]} maps to mode "
                        f"{command_mode}, expected {expected_mode}"
                    )

                trajectory_key = jax.random.fold_in(
                    jax.random.PRNGKey(seed), trajectory
                )
                reset_key, action_rng = jax.random.split(trajectory_key)
                state = reset_eval_state_with_command(env, reset_key, command)
                ever_terminated = False
                first_termination = -1

                for timestep in range(N_STEPS):
                    previous_observation = state.obs
                    action_rng, action_key = jax.random.split(action_rng)
                    action, _ = inference_fn(state.obs, action_key)
                    state = jit_step(state, action)
                    rewards = state.info["rewards"]
                    _verify_score_schema(rewards)

                    done_this_step = bool(_as_scalar(state.done))
                    if done_this_step and not ever_terminated:
                        ever_terminated = True
                        first_termination = timestep

                    # Rewards above correspond to this step and the fixed command.
                    # Restore that command before the next policy observation.
                    state = preserve_fixed_command(
                        env, state, command, previous_observation
                    )
                    state_mode = int(_as_scalar(state.info["mode"]))
                    if state_mode != expected_mode:
                        raise RuntimeError(
                            f"Fixed {gait} command changed mode at trajectory "
                            f"{trajectory}, timestep {timestep}: {state_mode}"
                        )
                    actual_command = np.asarray(
                        jax.device_get(state.info["command"]), dtype=np.float32
                    )
                    expected_command = np.asarray(
                        [sampled_vx[trajectory], 0.0, 0.0], dtype=np.float32
                    )
                    if not np.array_equal(actual_command, expected_command):
                        raise RuntimeError(
                            f"Command changed at trajectory {trajectory}, "
                            f"timestep {timestep}: {actual_command}"
                        )

                    values = {
                        key: float(_as_scalar(rewards[key]))
                        for key in ALL_RECORDED_METRIC_KEYS
                    }
                    arrays["trajectory"][trajectory, timestep] = trajectory
                    arrays["timestep"][trajectory, timestep] = timestep
                    arrays["command_vx"][trajectory, timestep] = sampled_vx[trajectory]
                    arrays["command_vy"][trajectory, timestep] = 0.0
                    arrays["command_yaw"][trajectory, timestep] = 0.0
                    arrays["mode"][trajectory, timestep] = state_mode
                    arrays["done_this_step"][trajectory, timestep] = done_this_step
                    arrays["trajectory_terminated"][
                        trajectory, timestep
                    ] = ever_terminated
                    arrays["first_termination_timestep"][
                        trajectory, timestep
                    ] = first_termination
                    for key, value in values.items():
                        arrays[key][trajectory, timestep] = value

                    writer.writerow(
                        {
                            "gait": gait,
                            "trajectory": trajectory,
                            "timestep": timestep,
                            "command_vx": float(sampled_vx[trajectory]),
                            "command_vy": 0.0,
                            "command_yaw": 0.0,
                            "mode": state_mode,
                            "mode_name": MODE_NAMES[state_mode],
                            "done_this_step": int(done_this_step),
                            "trajectory_terminated": int(ever_terminated),
                            "first_termination_timestep": first_termination,
                            "source_policy_path": str(checkpoint_path),
                            "source_policy_tree_sha256": checkpoint_hash,
                            "observation_layout": observation_metadata[
                                "observation_layout"
                            ],
                            "observation_dimension": observation_metadata[
                                "observation_dimension"
                            ],
                            "observation_history_frames": observation_metadata[
                                "history_frames"
                            ],
                            "observation_features_per_frame": observation_metadata[
                                "features_per_frame"
                            ],
                            "observation_mode_one_hot_included": int(
                                observation_metadata["mode_one_hot_included"]
                            ),
                            **values,
                        }
                    )
                    emitted_rows += 1
                    csv_file.flush()

                    if done_this_step:
                        enforce_no_termination(done_this_step, gait, trajectory, timestep)

        actual_rows = emitted_rows
        expected_rows = N_TRAJECTORIES * N_STEPS
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Emitted row count {actual_rows} does not equal {expected_rows}"
            )
        validate_recorded_arrays(arrays)

        npz_payload = dict(
            gait=np.asarray(gait),
            seed=np.asarray(seed, dtype=np.int64),
            source_policy_path=np.asarray(str(checkpoint_path)),
            source_policy_tree_sha256=np.asarray(checkpoint_hash),
            observation_layout=np.asarray(
                observation_metadata["observation_layout"]
            ),
            observation_dimension=np.asarray(
                observation_metadata["observation_dimension"], dtype=np.int32
            ),
            observation_history_frames=np.asarray(
                observation_metadata["history_frames"], dtype=np.int32
            ),
            observation_features_per_frame=np.asarray(
                observation_metadata["features_per_frame"], dtype=np.int32
            ),
            observation_mode_one_hot_included=np.asarray(
                observation_metadata["mode_one_hot_included"], dtype=np.bool_
            ),
            individual_score_keys=np.asarray(INDIVIDUAL_SCORE_KEYS),
            grouped_score_keys=np.asarray(GROUPED_SCORE_KEYS),
            diagnostic_metric_keys=np.asarray(DIAGNOSTIC_KEYS),
            sampled_vx=sampled_vx,
            **arrays,
        )
        npz_partial = _write_npz_atomic(npz_final, npz_payload)
        validate_csv_and_npz_match(csv_partial, npz_partial, metadata_path)
        csv_partial.replace(csv_final)
        npz_partial.replace(npz_final)

        metadata.update(
            {
                "status": "complete",
                "completed_rows": actual_rows,
                "trajectories_with_termination": int(
                    np.any(arrays["done_this_step"], axis=1).sum()
                ),
                "termination_events": int(arrays["done_this_step"].sum()),
                "sampled_vx_min": float(sampled_vx.min()),
                "sampled_vx_max": float(sampled_vx.max()),
                "csv_path": str(csv_final),
                "npz_path": str(npz_final),
                "csv_sha256": hashlib.sha256(csv_final.read_bytes()).hexdigest(),
                "npz_sha256": hashlib.sha256(npz_final.read_bytes()).hexdigest(),
            }
        )
        _write_json_atomic(metadata_path, metadata)
        return metadata
    except BaseException as error:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_csv_path": (
                    str(csv_partial) if csv_partial.exists() else None
                ),
                "partial_npz_path": (
                    str(npz_partial) if npz_partial.exists() else None
                ),
            }
        )
        _write_json_atomic(metadata_path, metadata)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw walk/trot expert STL robustness. No q50 or filtering "
            "is performed."
        )
    )
    parser.add_argument(
        "--gait",
        choices=("all", "walk", "trot"),
        default="all",
        help="Collect both gaits by default, or run either gait independently.",
    )
    parser.add_argument(
        "--walk-policy",
        type=Path,
        default=DEFAULT_WALK_POLICY,
        help=(
            "Walk expert checkpoint directory; PLACEHOLDER until supplied."
        ),
    )
    parser.add_argument(
        "--trot-policy",
        type=Path,
        default=DEFAULT_TROT_POLICY,
        help=(
            "Trot expert checkpoint directory; PLACEHOLDER until supplied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "collection" / "raw_data",
        help="New output root; each gait subdirectory must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load each policy/layout and run one step without collecting data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gaits = ("walk", "trot") if args.gait == "all" else (args.gait,)
    policies = {
        "walk": args.walk_policy,
        "trot": args.trot_policy,
    }
    for gait in gaits:
        checkpoint_path = policies[gait]
        if str(checkpoint_path) == "PLACEHOLDER" or not checkpoint_path.expanduser().is_dir():
            raise FileNotFoundError(
                f"PLACEHOLDER or missing {gait} expert checkpoint: supply --{gait}-policy."
            )
    reserved_outputs = None
    if not args.preflight_only:
        reserved_outputs = reserve_output_dirs(args.output_dir, gaits)
    results = [
        collect_gait(
            gait=gait,
            checkpoint_path=policies[gait],
            gait_output=None if reserved_outputs is None else reserved_outputs[gait],
            seed=args.seed,
            preflight_only=args.preflight_only,
        )
        for gait in gaits
    ]
    print(json.dumps({"results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
