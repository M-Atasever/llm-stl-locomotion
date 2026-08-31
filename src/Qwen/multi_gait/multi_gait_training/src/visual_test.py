#!/usr/bin/env python3
"""Render the three supplied fixed post-training velocity tests."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import jax
from jax import numpy as jp
import mediapy as media
import numpy as np
from brax import math
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from Barkour import BarkourEnv
from coeff_config import (
    validate_history_config,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    TROT_TO_BOUND_ENTER,
    WALK_TO_TROT_ENTER,
)


VELOCITIES = (0.5, 1.0, 1.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=2)
    return parser.parse_args()


def mode_from_command(command):
    vx = jp.abs(command[0])
    return jp.where(
        vx >= TROT_TO_BOUND_ENTER,
        jp.array(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx >= WALK_TO_TROT_ENTER,
            jp.array(MODE_TROT, dtype=jp.int32),
            jp.array(MODE_WALK, dtype=jp.int32),
        ),
    )


def reset_with_command(env, rng, command):
    state = env.reset(rng)
    info = dict(state.info)
    info["command"] = command
    info["mode"] = mode_from_command(command)
    info["history_len"] = jp.array(0, dtype=jp.int32)
    obs = env._get_obs(state.pipeline_state, info, jp.zeros_like(state.obs))
    return state.replace(obs=obs, info=info)


def load_policy(env, checkpoint_path):
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=0,
        episode_length=1000,
        normalize_observations=True,
        restore_checkpoint_path=checkpoint_path,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(128, 128, 128, 128),
        ),
    )
    return jax.jit(make_inference_fn(params))


def main() -> None:
    args = parse_args()
    validate_history_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = BarkourEnv(
        command_sampling="walk_trot_only", obs_noise=0.0, kick_vel=0.0
    )
    inference_fn = load_policy(env, args.ckpt_path)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    summaries = []

    for velocity in VELOCITIES:
        rng, reset_rng = jax.random.split(rng)
        command = jp.array([velocity, 0.0, 0.0], dtype=jp.float32)
        state = reset_with_command(env, reset_rng, command)
        trajectory = [state.pipeline_state]
        local_speeds = []
        terminated = False

        for _ in range(args.steps):
            rng, action_rng = jax.random.split(rng)
            action, _ = inference_fn(state.obs, action_rng)
            state = jit_step(state, action)
            local_velocity = math.rotate(
                state.pipeline_state.xd.vel[0],
                math.quat_inv(state.pipeline_state.x.rot[0]),
            )
            local_speeds.append(float(local_velocity[0]))
            terminated = terminated or bool(state.done)
            trajectory.append(state.pipeline_state)

        frames = env.render(
            trajectory[:: args.render_every],
            camera="track",
            width=480,
            height=360,
        )
        video_path = args.output_dir / f"velocity_{velocity:.1f}_mps.mp4"
        media.write_video(
            video_path,
            frames,
            fps=1.0 / env.dt / args.render_every,
        )
        summaries.append(
            {
                "command_velocity_mps": velocity,
                "mean_local_velocity_mps": float(np.mean(local_speeds)),
                "terminated": terminated,
                "video": video_path.name,
            }
        )
        print(video_path)

    (args.output_dir / "visual_test_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
