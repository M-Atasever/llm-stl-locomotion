"""Warm-start-only 400M stage-2 walk+trot+bound PPO launcher."""

from __future__ import annotations

import argparse
from datetime import datetime
import functools
import json
import os
from pathlib import Path
import shlex
import sys

# Keep runtime bytecode out of the final source directory.
sys.dont_write_bytecode = True

import jax
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from coeff_config import (
    DISABLED_GPT55_STL_THRESHOLDS,
    EARLY_TERMINATION_PENALTY,
    ENABLE_BOUND_GAIT,
    TRAINING_BOUND_STRAIGHT_COMMANDS,
    TRAINING_COMMAND_SAMPLING,
    TRAINING_GAIT_MODES,
    TRAINING_KICK_VEL,
    TRAINING_OBS_NOISE,
    TRAINING_NUM_TIMESTEPS,
)


def domain_randomize(sys, rng):
  """Randomizes friction and actuator gains, matching the supplied launcher."""

  @jax.vmap
  def randomize_one(key):
    _, key = jax.random.split(key, 2)
    friction_value = jax.random.uniform(
        key, (1,), minval=0.6, maxval=1.4
    )
    friction = sys.geom_friction.at[:, 0].set(friction_value)

    _, key = jax.random.split(key, 2)
    gain_delta = jax.random.uniform(key, (1,), minval=-5.0, maxval=5.0)
    gain_value = gain_delta + sys.actuator_gainprm[:, 0]
    gain = sys.actuator_gainprm.at[:, 0].set(gain_value)
    bias = sys.actuator_biasprm.at[:, 1].set(-gain_value)
    return friction, gain, bias

  friction, gain, bias = randomize_one(rng)
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


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_name", default=None)
  parser.add_argument(
      "--output_root",
      default=str(
          Path(__file__).resolve().parents[3] / "runs" / "multi_gait"
      ),
  )
  parser.add_argument(
      "--num_timesteps", type=int, default=TRAINING_NUM_TIMESTEPS
  )
  parser.add_argument("--num_envs", type=int, default=8192)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--selection_quantile",
      choices=("q50",),
      default="q50",
      help="Frozen q50 predicate selection used by this treatment.",
  )
  parser.add_argument(
      "--resume_checkpoint",
      required=True,
      help="Numeric Orbax checkpoint directory selected from stage 1.",
  )
  parser.add_argument(
      "--starting_checkpoint_steps",
      type=int,
      required=True,
      help="Stage-1 environment-step count encoded by the checkpoint name.",
  )
  parser.add_argument(
      "--early_termination_penalty",
      type=float,
      default=EARLY_TERMINATION_PENALTY,
      help="Unscaled cost applied only on physical-failure transitions.",
  )
  parser.add_argument(
      "--smoke_test",
      action="store_true",
      help="Use a small run to validate compilation and data flow.",
  )
  return parser.parse_args()


def json_value(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return str(value)


def main():
  args = parse_args()
  os.environ["STL_SELECTION_QUANTILE"] = args.selection_quantile
  from barkour import BarkourEnv
  from stl_reward import (
      ACTIVE_SPECIFICATIONS_BY_SCOPE,
      BOUND_Q50_EXCLUDED_SPECIFICATIONS,
      BOUND_Q50_SELECTED_SPECIFICATIONS,
      EXCLUDED_SPECIFICATIONS,
      SELECTION_QUANTILE,
      _REQUIRED_STL_PARAMS,
  )

  if SELECTION_QUANTILE != args.selection_quantile:
    raise RuntimeError(
        "STL selection was imported before --selection_quantile took effect."
    )
  if tuple(ACTIVE_SPECIFICATIONS_BY_SCOPE["bound"]) != (
      BOUND_Q50_SELECTED_SPECIFICATIONS
  ):
    raise RuntimeError("BOUND q50 predicate selection is not frozen correctly.")
  if args.early_termination_penalty < 0:
    raise ValueError("--early_termination_penalty must be nonnegative.")
  if not args.smoke_test and args.num_timesteps != TRAINING_NUM_TIMESTEPS:
    raise ValueError(
        "stage 2 requires exactly 400M additional steps; use --smoke_test "
        "for a compilation-only run"
    )
  if args.starting_checkpoint_steps <= 0:
    raise ValueError("--starting_checkpoint_steps must be positive.")
  resume_checkpoint = Path(args.resume_checkpoint).expanduser().resolve()
  if not resume_checkpoint.is_dir():
    raise FileNotFoundError(
        f"--resume_checkpoint is not a directory: {resume_checkpoint}"
    )
  if not resume_checkpoint.name.isdigit():
    raise ValueError(
        "--resume_checkpoint must end in its numeric environment-step label"
    )
  if int(resume_checkpoint.name) != args.starting_checkpoint_steps:
    raise ValueError(
        "--starting_checkpoint_steps does not match the checkpoint directory "
        f"name: {args.starting_checkpoint_steps} != {resume_checkpoint.name}"
    )
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  requested_label = "smoke" if args.smoke_test else f"{args.num_timesteps}_steps"
  run_name = (
      args.run_name
      or (
          "phase1_final_multi_gait_"
          f"{requested_label}_seed_{args.seed}_{timestamp}"
      )
  )
  if (
      not run_name
      or run_name in {".", ".."}
      or Path(run_name).name != run_name
  ):
    raise ValueError("--run_name must be one plain directory name")
  output_root = Path(args.output_root).expanduser().resolve()
  package_root = Path(__file__).resolve().parent.parent
  if output_root == package_root or package_root in output_root.parents:
    raise ValueError(
        "--output_root must be outside the final source package"
    )
  if output_root.exists() and not output_root.is_dir():
    raise ValueError(f"--output_root is not a directory: {output_root}")
  run_dir = output_root / run_name
  try:
    run_dir.mkdir(parents=True, exist_ok=False)
  except FileExistsError as exc:
    raise FileExistsError(
        f"refusing to reuse existing run directory: {run_dir}"
    ) from exc
  checkpoint_dir = run_dir / "checkpoints"
  checkpoint_dir.mkdir()

  num_timesteps = args.num_timesteps
  num_envs = args.num_envs
  num_evals = 10
  if args.smoke_test:
    num_timesteps = min(num_timesteps, 65_536)
    num_envs = min(num_envs, 128)
    num_evals = 2

  steps_per_update = 256 * 30 * 32 * 1
  evaluation_intervals = max(num_evals - 1, 1)
  steps_per_interval = steps_per_update * evaluation_intervals
  expected_stage_steps = (
      (num_timesteps + steps_per_interval - 1) // steps_per_interval
  ) * steps_per_interval

  metadata = {
      "experiment": "phase1_final_multi_gait",
      "variant": "Multi-gait warm start with expert-selected q50 predicates",
      "created_at": datetime.now().isoformat(),
      "command": " ".join(shlex.quote(value) for value in sys.argv),
      "num_timesteps": num_timesteps,
      "additional_num_timesteps": num_timesteps,
      "starting_checkpoint_steps": args.starting_checkpoint_steps,
      "cumulative_target_steps": args.starting_checkpoint_steps + num_timesteps,
      "expected_realized_additional_timesteps": expected_stage_steps,
      "expected_final_cumulative_steps": (
          args.starting_checkpoint_steps + expected_stage_steps
      ),
      "step_rounding": (
          "Brax PPO rounds the requested budget up to complete evaluation "
          "intervals; final_metrics records the observed stage step count"
      ),
      "num_envs": num_envs,
      "seed": args.seed,
      "selection_quantile": args.selection_quantile,
      "command_sampling": TRAINING_COMMAND_SAMPLING,
      "training_gait_mode_ids": TRAINING_GAIT_MODES,
      "bound_enabled": ENABLE_BOUND_GAIT,
      "smoke_test": args.smoke_test,
      "resume_checkpoint": str(resume_checkpoint),
      "curriculum_stage": "walk_trot_bound_joint",
      "warm_start_only": True,
      "bound_command_configuration": {
          "sample_range_mps": [1.55, 1.9],
          "straight_commands": TRAINING_BOUND_STRAIGHT_COMMANDS,
          "observation_noise": TRAINING_OBS_NOISE,
          "kick_velocity": TRAINING_KICK_VEL,
      },
      "checkpoint_step_labels": (
          "cumulative = starting_checkpoint_steps + stage_steps"
      ),
      "llm_specification": {
          "response_path": "stl_specifications.json",
          "model": "gpt-5.5-2026-04-23",
      },
      "velocity_tracking_treatment": {
          "velocity_error": (
              "abs(vx_cmd-vx)/max(abs(vx_cmd),0.10) + "
              "0.01*abs(vy_cmd-vy)/0.05 + "
              "0.01*abs(yaw_cmd-yaw_rate)/0.05"
          ),
          "rho_tracking_atomic": "1 - velocity_error",
          "temporal_operator": "causal minimum",
          "tracking_weight": 1.0,
          "e_vel_max": 1.0,
          "predicate_refiltered": True,
      },
      "required_stl_parameters": _REQUIRED_STL_PARAMS,
      "active_stl_specifications": ACTIVE_SPECIFICATIONS_BY_SCOPE,
      "excluded_stl_specifications": EXCLUDED_SPECIFICATIONS,
      "selection_analysis": {
          "scope": "eight BOUND-gait predicates only",
          "num_trajectories": 50,
          "steps_per_trajectory": 500,
          "warmup_timesteps_ignored": 50,
          "post_warmup_samples_per_predicate": 22500,
          "rule": (
              "keep a BOUND predicate when post-warmup q50 >= 0; "
              "exclude when q50 < 0; exact zero is kept"
          ),
          "selected_bound_predicates": BOUND_Q50_SELECTED_SPECIFICATIONS,
          "excluded_bound_predicates": BOUND_Q50_EXCLUDED_SPECIFICATIONS,
          "manual_overrides": [],
          "walk_trot_selection": "selected q50 predicate set",
      },
      "disabled_stl_thresholds": DISABLED_GPT55_STL_THRESHOLDS,
      "early_termination_penalty": {
          "magnitude": args.early_termination_penalty,
          "application": (
              "unscaled post-dt cost on physical-failure done transitions"
          ),
          "symbolic_stl_predicate": False,
          "reason": (
              "prevents the policy from maximizing return by terminating "
              "early while the STL reward is negative"
          ),
      },
      "visual_evaluation_speeds_mps": [0.3, 0.5, 0.9, 1.0, 1.2],
  }
  (run_dir / "run_metadata.json").write_text(
      json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
  )

  def policy_params_fn(current_step, make_policy, params):
    del make_policy
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    cumulative_step = args.starting_checkpoint_steps + int(current_step)
    checkpointer.save(
        checkpoint_dir / str(cumulative_step),
        params,
        force=True,
        save_args=save_args,
    )

  metrics_path = run_dir / "metrics.jsonl"
  last_stage_steps = 0

  def progress(num_steps, metrics):
    nonlocal last_stage_steps
    last_stage_steps = int(num_steps)
    cumulative_steps = args.starting_checkpoint_steps + last_stage_steps
    record = {
        "num_steps": cumulative_steps,
        "cumulative_steps": cumulative_steps,
        "stage_steps": last_stage_steps,
        "starting_checkpoint_steps": args.starting_checkpoint_steps,
        **{key: json_value(value) for key, value in metrics.items()},
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        f"stage_steps={last_stage_steps} cumulative_steps={cumulative_steps} "
        f"reward={record.get('eval/episode_reward', 'unavailable')}"
    )

  make_networks_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=(128, 128, 128, 128),
  )
  train_fn = functools.partial(
      ppo.train,
      num_timesteps=num_timesteps,
      num_evals=num_evals,
      reward_scaling=1,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=30,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.955,
      learning_rate=0.00015,
      entropy_cost=0.004,
      num_envs=num_envs,
      batch_size=256,
      network_factory=make_networks_factory,
      randomization_fn=domain_randomize,
      policy_params_fn=policy_params_fn,
      seed=args.seed,
  )
  train_fn = functools.partial(
      train_fn, restore_checkpoint_path=str(resume_checkpoint)
  )

  environment = BarkourEnv(
      command_sampling=TRAINING_COMMAND_SAMPLING,
      bound_straight_commands=TRAINING_BOUND_STRAIGHT_COMMANDS,
      obs_noise=TRAINING_OBS_NOISE,
      kick_vel=TRAINING_KICK_VEL,
      early_termination_penalty=args.early_termination_penalty,
  )
  evaluation_environment = BarkourEnv(
      command_sampling=TRAINING_COMMAND_SAMPLING,
      bound_straight_commands=TRAINING_BOUND_STRAIGHT_COMMANDS,
      obs_noise=TRAINING_OBS_NOISE,
      kick_vel=TRAINING_KICK_VEL,
      early_termination_penalty=args.early_termination_penalty,
  )
  _, params, final_metrics = train_fn(
      environment=environment,
      progress_fn=progress,
      eval_env=evaluation_environment,
  )
  model.save_params(run_dir / "final_policy", params)
  final_metric_record = {
      "num_steps": args.starting_checkpoint_steps + last_stage_steps,
      "cumulative_steps": args.starting_checkpoint_steps + last_stage_steps,
      "stage_steps": last_stage_steps,
      "starting_checkpoint_steps": args.starting_checkpoint_steps,
      **{key: json_value(value) for key, value in final_metrics.items()},
  }
  (run_dir / "final_metrics.json").write_text(
      json.dumps(
          final_metric_record,
          indent=2,
          sort_keys=True,
      )
      + "\n",
      encoding="utf-8",
  )
  print(f"Training complete. Artifacts: {run_dir}")


if __name__ == "__main__":
  main()
