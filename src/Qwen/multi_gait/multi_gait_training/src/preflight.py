#!/usr/bin/env python3
"""Preflight the multi-gait walk/trot training."""

import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import jax
from jax import numpy as jp

from Barkour import BARKOUR_ROOT_PATH, BarkourEnv
from coeff_config import (
    K_REQUIRE_3PLUS, H_by_mode, MODE_BOUND, MODE_TROT, MODE_WALK,
    validate_history_config,
)
from stl_reward import reward_step


def synthetic_input(vx: float, vy: float, yaw: float, slip: float = 0.0):
    h = 30
    contacts = jp.ones((h, 4), dtype=jp.float32)
    return {
        "tau_history": jp.zeros((h, 12)),
        "contact_history": contacts,
        "feet_history": jp.zeros((h, 4, 2)),
        "CoM_history": jp.zeros((h, 2)),
        "lin_velocity_history": jp.tile(jp.array([vx, vy, 0.0]), (h, 1)),
        "ang_velocity_history": jp.full((h,), yaw),
        "com_z_history": jp.full((h,), 0.26),
        "roll_history": jp.zeros(h),
        "pitch_history": jp.zeros(h),
        "slipmax_history": jp.full((h,), slip),
    }


def main():
    validate_history_config()
    model_path = Path(BARKOUR_ROOT_PATH) / "scene_mjx.xml"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path}")

    assert K_REQUIRE_3PLUS / H_by_mode[MODE_WALK] == 11 / 30
    command = jp.array([0.5, 0.0, 0.0])
    baseline = reward_step(synthetic_input(0.4, 0.0, 0.0), command, MODE_WALK, 30)
    lateral_yaw_changed = reward_step(
        synthetic_input(0.4, 8.0, 8.0), command, MODE_WALK, 30
    )
    assert len(baseline) == 43
    assert bool(jp.isclose(baseline[26], lateral_yaw_changed[26]))

    walk_slip = reward_step(
        synthetic_input(0.4, 0.0, 0.0, slip=10.0), command, MODE_WALK, 30
    )
    trot_command = jp.array([1.0, 0.0, 0.0])
    trot_clean = reward_step(
        synthetic_input(1.0, 0.0, 0.0), trot_command, MODE_TROT, 30
    )
    trot_slip = reward_step(
        synthetic_input(1.0, 0.0, 0.0, slip=10.0),
        trot_command,
        MODE_TROT,
        30,
    )
    assert bool(jp.isclose(baseline[2], walk_slip[2]))
    assert not bool(jp.isclose(trot_clean[2], trot_slip[2]))

    env = BarkourEnv(command_sampling="walk_trot_only", obs_noise=0.0, kick_vel=0.0)
    state = env.reset(jax.random.PRNGKey(0))
    next_state = jax.jit(env.step)(state, jp.zeros(12))
    jax.block_until_ready(next_state.reward)
    assert state.obs.shape == (510,)
    assert next_state.obs.shape == (510,)
    keys = jax.random.split(jax.random.PRNGKey(1), 512)
    modes = jax.vmap(env._mode_from_command)(jax.vmap(env.sample_command)(keys))
    assert not bool(jp.any(modes == MODE_BOUND))

    print(f"JAX backend: {jax.default_backend()}")
    print("Environment reset/JIT step: PASS")
    print("multi-gait reward tuple: PASS")
    print("Forward-velocity-only tracking: PASS")
    print("q50 mode-specific slip selection: PASS")
    print("p_3plus fraction 11/30: PASS")
    print("Walk/trot-only sampling: PASS")


if __name__ == "__main__":
    main()
