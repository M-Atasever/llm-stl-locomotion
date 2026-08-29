"""Final eight-stage, three-regime gait-agnostic STL curriculum.

Stages, 50M requested steps each:
  1. vx ~ U(0.0, 0.5)
  2. vx ~ U(0.0, 0.75)
  3. vx ~ U(0.0, 1.0)
  4. vx ~ U(0.0, 1.2)
  5. vx ~ U(0.0, 1.4)
  6. vx ~ U(0.0, 1.6)
  7. vx ~ U(0.0, 1.8)
  8. vx ~ U(0.0, 1.9)

Each stage samples reachable gaits using the configured walk/trot/bound regime
probabilities, then samples velocity inside that gait's part of the stage
range. Each stage initializes its policy, value function, and observation
normalizer from the previous stage's final parameters via Brax
``restore_params``. PPO reinitializes optimizer state at a stage boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import shlex
import sys
import traceback

sys.dont_write_bytecode = True
os.environ.setdefault(
    "MUJOCO_GL", "glfw" if sys.platform == "darwin" else "disable"
)
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
  os.environ["XLA_FLAGS"] = (
      _xla_flags + " --xla_gpu_triton_gemm_any=True"
  ).strip()

import jax
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from barkour import BARKOUR_ROOT_PATH, BarkourEnv
from coeff_config import (
    BOUND_ACTIVE_PREDICATE_MASK,
    BOUND_EXCLUDED_PREDICATES,
    BOUND_Q50_BASELINE_PREDICATE_MASK,
    BOUND_Q50_ROBUSTNESS,
    BOUND_TO_TROT_EXIT,
    CURRICULUM_USES_REGIME_SAMPLE_PROBS,
    CURRICULUM_STAGES,
    EARLY_TERMINATION_PENALTY,
    FORWARD_TRACKING_REWARD_WEIGHT,
    H,
    LATERAL_PREDICATE_INDEX,
    LATERAL_Q50_OVERRIDE_ENABLED,
    LATERAL_Q50_OVERRIDE_REASON,
    LATERAL_TRACKING_REWARD_WEIGHT,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    PREDICATE_REWARD_WEIGHTS,
    Q50_ANALYSIS_EPS_VX,
    Q50_ANALYSIS_EPS_VY,
    Q50_ANALYSIS_EPS_WZ,
    Q50_ANALYSIS_N_CONTACT_MIN,
    Q50_ANALYSIS_THETA_MAX_DEGREES,
    Q50_ANALYSIS_Z_MIN,
    Q50_FILTER_DROP_PREFIX_TIMESTEPS,
    Q50_FILTER_OBSERVATIONS_PER_GAIT,
    STL_PREDICATE_ORDER,
    TRACKING_AUX_TO_FORWARD_WEIGHT_RATIO,
    TRAINING_NUM_ENVS,
    TROT_TO_BOUND_ENTER,
    TROT_ACTIVE_PREDICATE_MASK,
    TROT_EXCLUDED_PREDICATES,
    TROT_Q50_BASELINE_PREDICATE_MASK,
    TROT_Q50_ROBUSTNESS,
    VISUAL_EVAL_SPEEDS,
    WALK_ACTIVE_PREDICATE_MASK,
    WALK_EXCLUDED_PREDICATES,
    WALK_Q50_BASELINE_PREDICATE_MASK,
    WALK_Q50_ROBUSTNESS,
    WALK_TO_TROT_ENTER,
    YAW_PREDICATE_INDEX,
    YAW_Q50_OVERRIDE_ENABLED,
    YAW_Q50_OVERRIDE_REASON,
    beta,
    bound_vx_sample_range,
    eps_vx,
    eps_vy,
    eps_wz,
    n_contact_min,
    regime_sample_probs,
    theta_max,
    YAW_TRACKING_REWARD_WEIGHT,
    z_min,
)


POLICY_HIDDEN_LAYER_SIZES = (128, 128, 128, 128)
NUM_EVALS_PER_STAGE = 10
BATCH_SIZE = 256
UNROLL_LENGTH = 30
NUM_MINIBATCHES = 32
ACTION_REPEAT = 1


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


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


def parse_args() -> argparse.Namespace:
  package_root = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run_name", default=None)
  parser.add_argument(
      "--output_root",
      default=os.environ.get("BARKOUR_OUTPUT_ROOT", str(package_root / "runs")),
  )
  parser.add_argument("--num_envs", type=int, default=TRAINING_NUM_ENVS)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--start_stage",
      type=int,
      default=1,
      choices=range(1, len(CURRICULUM_STAGES) + 1),
      help="First curriculum stage to run (1-indexed).",
  )
  parser.add_argument(
      "--restore_from",
      default=None,
      help=(
          "Path to a saved final_policy params file to initialize the first "
          "executed stage from. Required when --start_stage > 1."
      ),
  )
  parser.add_argument(
      "--early_termination_penalty",
      type=float,
      default=EARLY_TERMINATION_PENALTY,
  )
  parser.add_argument(
      "--smoke_test",
      action="store_true",
      help="Compile and exercise all eight stages with tiny disposable budgets.",
  )
  return parser.parse_args()


def json_value(value):
  array = np.asarray(jax.device_get(value))
  if array.size == 1:
    return float(array.reshape(()))
  return array.tolist()


def write_json(path: Path, value) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )


def expected_realized_steps(num_timesteps: int, num_evals: int) -> int:
  steps_per_training_step = (
      BATCH_SIZE * UNROLL_LENGTH * NUM_MINIBATCHES * ACTION_REPEAT
  )
  intervals = max(num_evals - 1, 1)
  steps_per_interval = steps_per_training_step * intervals
  return (
      (num_timesteps + steps_per_interval - 1) // steps_per_interval
  ) * steps_per_interval


def stage_gaits(vx_hi: float) -> list:
  gaits = ["walk"]
  if vx_hi > WALK_TO_TROT_ENTER:
    gaits.append("trot")
  if vx_hi > TROT_TO_BOUND_ENTER:
    gaits.append("bound")
  return gaits


def stage_gait_probabilities(vx_hi: float) -> dict:
  """Returns the renormalized gait mixture used by a curriculum stage."""
  names = ("walk", "trot", "bound")
  reachable = (
      True,
      vx_hi > WALK_TO_TROT_ENTER,
      vx_hi > TROT_TO_BOUND_ENTER,
  )
  raw = [
      probability if is_reachable else 0.0
      for probability, is_reachable in zip(
          regime_sample_probs, reachable, strict=True
      )
  ]
  normalizer = sum(raw)
  return {
      name: probability / normalizer
      for name, probability in zip(names, raw, strict=True)
      if probability > 0.0
  }


def format_speed_for_path(value: float) -> str:
  """Keeps non-tenth curriculum boundaries such as 0.75 exact in paths."""
  return f"{float(value):g}"


def validate_curriculum() -> None:
  """Prevents accidental drift from the final eight-stage curriculum."""
  actual = tuple(
      (stage["stage"], tuple(stage["vx_range"]), stage["num_timesteps"])
      for stage in CURRICULUM_STAGES
  )
  expected = (
      (1, (0.0, 0.5), 50_000_000),
      (2, (0.0, 0.75), 50_000_000),
      (3, (0.0, 1.0), 50_000_000),
      (4, (0.0, 1.2), 50_000_000),
      (5, (0.0, 1.4), 50_000_000),
      (6, (0.0, 1.6), 50_000_000),
      (7, (0.0, 1.8), 50_000_000),
      (8, (0.0, 1.9), 50_000_000),
  )
  if actual != expected:
    raise ValueError(
        "CURRICULUM_STAGES no longer matches the requested 8 x 50M plan: "
        f"{actual}"
    )
  if eps_vx != 0.25:
    raise ValueError(f"The final eps_vx must remain 0.25; got {eps_vx}.")
  active_thresholds = (
      TROT_TO_BOUND_ENTER,
      BOUND_TO_TROT_EXIT,
      tuple(bound_vx_sample_range),
  )
  expected_thresholds = (
      1.65,
      1.60,
      (1.65, 1.9),
  )
  if active_thresholds != expected_thresholds:
    raise ValueError(
        f"The final gait thresholds have drifted: {active_thresholds}"
    )
  if stage_gaits(1.6) != ["walk", "trot"]:
    raise ValueError("Stage 6 must remain walk/trot-only at the 1.65 boundary.")
  if stage_gaits(1.8) != ["walk", "trot", "bound"]:
    raise ValueError("Stage 7 must introduce bound commands.")
  if not CURRICULUM_USES_REGIME_SAMPLE_PROBS:
    raise ValueError("The curriculum must use gait-stratified sampling.")
  if tuple(regime_sample_probs) != (0.25, 0.35, 0.40):
    raise ValueError(
        f"Unexpected walk/trot/bound probabilities: {regime_sample_probs}"
    )
  if tuple(PREDICATE_REWARD_WEIGHTS) != (1.0, 0.1, 0.1, 1.0, 1.0, 1.0):
    raise ValueError(
        f"Unexpected predicate reward weights: {PREDICATE_REWARD_WEIGHTS}"
    )


def run_stage(
    args: argparse.Namespace,
    run_dir: Path,
    stage: dict,
    restore_params,
    curriculum_status: dict,
    curriculum_status_path: Path,
):
  """Trains one curriculum stage and returns its final (norm, policy, value)."""
  stage_id = stage["stage"]
  vx_range = tuple(stage["vx_range"])
  num_timesteps = stage["num_timesteps"]
  num_envs = args.num_envs
  num_evals = NUM_EVALS_PER_STAGE
  if args.smoke_test:
    num_timesteps = 65_536
    num_envs = min(num_envs, 128)
    num_evals = 2
  stage_seed = args.seed + stage_id - 1

  speed_lo = format_speed_for_path(vx_range[0])
  speed_hi = format_speed_for_path(vx_range[1])
  stage_dir = run_dir / f"stage{stage_id}_vx_{speed_lo}_{speed_hi}"
  stage_dir.mkdir(exist_ok=False)
  checkpoint_dir = stage_dir / "checkpoints"
  checkpoint_dir.mkdir()

  realized_steps = expected_realized_steps(num_timesteps, num_evals)
  metadata = {
      "experiment": "phase1_final_gait_agnostic",
      "created_at": utc_now(),
      "command": " ".join(shlex.quote(value) for value in sys.argv),
      "stage": stage_id,
      "total_stages": len(CURRICULUM_STAGES),
      "stage_directory": str(stage_dir),
      "curriculum_vx_range": list(vx_range),
      "reachable_gaits": stage_gaits(vx_range[1]),
      "gait_sampling_probabilities": stage_gait_probabilities(vx_range[1]),
      "initialized_from_previous_stage": restore_params is not None,
      "optimizer_state_restored": False,
      "requested_num_timesteps": num_timesteps,
      "expected_realized_num_timesteps": realized_steps,
      "step_rounding": "Brax PPO rounds up to complete evaluation intervals.",
      "num_envs": num_envs,
      "seed": stage_seed,
      "smoke_test": args.smoke_test,
      "command_sampling": "gait-stratified curriculum",
      "base_walk_trot_bound_probabilities": list(regime_sample_probs),
      "gait_thresholds": {
          "trot_to_bound_enter_active_mps": TROT_TO_BOUND_ENTER,
          "bound_to_trot_exit_active_mps": BOUND_TO_TROT_EXIT,
          "bound_vx_sample_range_active_mps": list(bound_vx_sample_range),
          "effect": (
              "Stage 6 remains walk/trot-only; stages 7-8 include bound."
          ),
      },
      "barkour_assets": str(BARKOUR_ROOT_PATH),
      "jax_devices": [str(device) for device in jax.devices()],
      "stl": {
          "formulas_by_gait": {
              "walk": (
                  "G_[0,H](e_vx <= eps_vx AND e_vy <= eps_vy AND "
                  "e_wz <= eps_wz AND "
                  "com_z >= z_min AND "
                  "max(abs(roll), abs(pitch)) <= theta_max AND "
                  "n_stance_contacts >= n_contact_min)"
              ),
              "trot": (
                  "G_[0,H](e_vx <= eps_vx AND e_vy <= eps_vy AND "
                  "e_wz <= eps_wz AND "
                  "com_z >= z_min AND "
                  "max(abs(roll), abs(pitch)) <= theta_max AND "
                  "n_stance_contacts >= n_contact_min)"
              ),
              "bound": (
                  "G_[0,H](e_vx <= eps_vx AND e_vy <= eps_vy AND "
                  "e_wz <= eps_wz AND "
                  "com_z >= z_min AND "
                  "max(abs(roll), abs(pitch)) <= theta_max AND "
                  "n_stance_contacts >= n_contact_min)"
              ),
          },
          "predicate_order": list(STL_PREDICATE_ORDER),
          "active_predicate_masks": {
              "walk": list(WALK_ACTIVE_PREDICATE_MASK),
              "trot": list(TROT_ACTIVE_PREDICATE_MASK),
              "bound": list(BOUND_ACTIVE_PREDICATE_MASK),
          },
          "excluded_predicates": {
              "walk": list(WALK_EXCLUDED_PREDICATES),
              "trot": list(TROT_EXCLUDED_PREDICATES),
              "bound": list(BOUND_EXCLUDED_PREDICATES),
          },
          "q50_filter": {
              "dropped_prefix_timesteps_per_trajectory": (
                  Q50_FILTER_DROP_PREFIX_TIMESTEPS
              ),
              "observations_per_expert_gait": (
                  Q50_FILTER_OBSERVATIONS_PER_GAIT
              ),
              "baseline_rule": (
                  "keep when gait-specific q50 robustness >= 0"
              ),
              "baseline_masks": {
                  "walk": list(WALK_Q50_BASELINE_PREDICATE_MASK),
                  "trot": list(TROT_Q50_BASELINE_PREDICATE_MASK),
                  "bound": list(BOUND_Q50_BASELINE_PREDICATE_MASK),
              },
              "raw_q50_robustness": {
                  "walk": list(WALK_Q50_ROBUSTNESS),
                  "trot": list(TROT_Q50_ROBUSTNESS),
                  "bound": list(BOUND_Q50_ROBUSTNESS),
              },
              "common_rescoring_thresholds": {
                  "eps_vx": Q50_ANALYSIS_EPS_VX,
                  "eps_vy": Q50_ANALYSIS_EPS_VY,
                  "eps_wz": Q50_ANALYSIS_EPS_WZ,
                  "z_min": Q50_ANALYSIS_Z_MIN,
                  "theta_max_degrees": Q50_ANALYSIS_THETA_MAX_DEGREES,
                  "n_contact_min": Q50_ANALYSIS_N_CONTACT_MIN,
              },
              "deliberate_overrides": {
                  "lateral_velocity_tracking": {
                      "enabled": LATERAL_Q50_OVERRIDE_ENABLED,
                      "predicate_index": LATERAL_PREDICATE_INDEX,
                      "walk_q50": WALK_Q50_ROBUSTNESS[
                          LATERAL_PREDICATE_INDEX
                      ],
                      "trot_q50": TROT_Q50_ROBUSTNESS[
                          LATERAL_PREDICATE_INDEX
                      ],
                      "bound_q50": BOUND_Q50_ROBUSTNESS[
                          LATERAL_PREDICATE_INDEX
                      ],
                      "eps_vy_mps": eps_vy,
                      "reward_weight": LATERAL_TRACKING_REWARD_WEIGHT,
                      "reason": LATERAL_Q50_OVERRIDE_REASON,
                  },
                  "yaw_rate_tracking": {
                      "enabled": YAW_Q50_OVERRIDE_ENABLED,
                      "predicate_index": YAW_PREDICATE_INDEX,
                      "walk_q50": WALK_Q50_ROBUSTNESS[
                          YAW_PREDICATE_INDEX
                      ],
                      "trot_q50": TROT_Q50_ROBUSTNESS[
                          YAW_PREDICATE_INDEX
                      ],
                      "bound_q50": BOUND_Q50_ROBUSTNESS[
                          YAW_PREDICATE_INDEX
                      ],
                      "eps_wz_rad_per_s": eps_wz,
                      "reward_weight": YAW_TRACKING_REWARD_WEIGHT,
                      "reason": YAW_Q50_OVERRIDE_REASON,
                  }
              },
          },
          "bound_expert_source": (
              "Real supplied bound expert; 50 trajectories x 500 timesteps, "
              "with timesteps 0..49 discarded before q50 selection."
          ),
          "dense_reward_predicate_weights": {
              "predicate_order": list(STL_PREDICATE_ORDER),
              "weights": list(PREDICATE_REWARD_WEIGHTS),
              "forward_tracking": FORWARD_TRACKING_REWARD_WEIGHT,
              "lateral_tracking": LATERAL_TRACKING_REWARD_WEIGHT,
              "yaw_tracking": YAW_TRACKING_REWARD_WEIGHT,
              "auxiliary_to_forward_ratio": (
                  TRACKING_AUX_TO_FORWARD_WEIGHT_RATIO
              ),
              "exact_stl_robustness_is_unweighted": True,
          },
          "H": H,
          "beta": beta,
          "eps_vx": eps_vx,
          "eps_vy": eps_vy,
          "eps_wz": eps_wz,
          "z_min": z_min,
          "theta_max_radians": theta_max,
          "n_contact_min": n_contact_min,
          "z_max": "disabled",
          "n_contact_max": "disabled",
          "note": (
              "Thresholds are identical across all curriculum stages; only "
              "the command distribution changes. The baseline q50 gate is "
              "gait-conditioned from real walk, trot, and bound experts; "
              "lateral and yaw are deliberately active at 0.1 importance "
              "relative to forward tracking."
          ),
      },
      "early_termination_penalty": {
          "magnitude": args.early_termination_penalty,
          "is_stl_predicate": False,
      },
      "ppo": {
          "num_evals": num_evals,
          "episode_length": 1000,
          "unroll_length": UNROLL_LENGTH,
          "num_minibatches": NUM_MINIBATCHES,
          "num_updates_per_batch": 4,
          "discounting": 0.955,
          "learning_rate": 0.00015,
          "entropy_cost": 0.004,
          "batch_size": BATCH_SIZE,
          "policy_hidden_layer_sizes": list(POLICY_HIDDEN_LAYER_SIZES),
      },
      "visual_evaluation_speeds_mps": list(VISUAL_EVAL_SPEEDS),
  }
  write_json(stage_dir / "run_metadata.json", metadata)

  status_path = stage_dir / "status.json"
  status = {
      "state": "initializing",
      "updated_at": utc_now(),
      "last_step": 0,
      "stage": stage_id,
      "stage_directory": str(stage_dir),
  }
  write_json(status_path, status)

  checkpointer = ocp.PyTreeCheckpointer()

  def policy_params_fn(current_step, make_policy, params):
    del make_policy
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(
        checkpoint_dir / str(int(current_step)),
        params,
        force=True,
        save_args=save_args,
    )

  metrics_path = stage_dir / "metrics.jsonl"
  last_step = 0

  def progress(num_steps, metrics):
    nonlocal last_step
    last_step = int(num_steps)
    record = {
        "num_steps": last_step,
        "recorded_at": utc_now(),
        **{key: json_value(value) for key, value in metrics.items()},
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(record, sort_keys=True) + "\n")
    status.update(
        state="training",
        updated_at=record["recorded_at"],
        last_step=last_step,
        latest_eval_episode_reward=record.get("eval/episode_reward"),
    )
    write_json(status_path, status)
    curriculum_status.update(
        updated_at=record["recorded_at"],
        current_stage=stage_id,
        current_stage_step=last_step,
    )
    write_json(curriculum_status_path, curriculum_status)
    print(
        f"stage={stage_id} steps={last_step} "
        f"reward={record.get('eval/episode_reward', 'unavailable')}",
        flush=True,
    )

  make_networks_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=POLICY_HIDDEN_LAYER_SIZES,
  )
  train_fn = functools.partial(
      ppo.train,
      num_timesteps=num_timesteps,
      num_evals=num_evals,
      reward_scaling=1,
      episode_length=1000,
      normalize_observations=True,
      action_repeat=ACTION_REPEAT,
      unroll_length=UNROLL_LENGTH,
      num_minibatches=NUM_MINIBATCHES,
      num_updates_per_batch=4,
      discounting=0.955,
      learning_rate=0.00015,
      entropy_cost=0.004,
      num_envs=num_envs,
      batch_size=BATCH_SIZE,
      network_factory=make_networks_factory,
      randomization_fn=domain_randomize,
      policy_params_fn=policy_params_fn,
      seed=stage_seed,
      restore_params=restore_params,
      restore_value_fn=True,
  )

  try:
    environment = BarkourEnv(
        command_sampling="curriculum",
        curriculum_vx_range=vx_range,
        early_termination_penalty=args.early_termination_penalty,
    )
    evaluation_environment = BarkourEnv(
        command_sampling="curriculum",
        curriculum_vx_range=vx_range,
        early_termination_penalty=args.early_termination_penalty,
    )
    _, params, final_metrics = train_fn(
        environment=environment,
        progress_fn=progress,
        eval_env=evaluation_environment,
    )
    final_policy_path = stage_dir / "final_policy"
    model.save_params(final_policy_path, params)
    write_json(
        stage_dir / "final_metrics.json",
        {
            "observed_final_step": last_step,
            **{key: json_value(value) for key, value in final_metrics.items()},
        },
    )
    status.update(
        state="completed",
        updated_at=utc_now(),
        last_step=last_step,
        final_policy=str(final_policy_path),
    )
    write_json(status_path, status)
    print(f"Stage {stage_id} complete. Artifacts: {stage_dir}", flush=True)
    return params, final_policy_path
  except BaseException as exc:
    status.update(
        state="failed",
        updated_at=utc_now(),
        last_step=last_step,
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
    )
    write_json(status_path, status)
    raise


def main() -> None:
  args = parse_args()
  validate_curriculum()
  if args.early_termination_penalty < 0:
    raise ValueError("--early_termination_penalty must be nonnegative.")
  if args.num_envs <= 0:
    raise ValueError("--num_envs must be positive.")
  if args.start_stage > 1 and args.restore_from is None:
    raise ValueError(
        "--restore_from is required when starting at a later stage, so the "
        "policy continues from the previous stage's final parameters."
    )
  if args.start_stage == 1 and args.restore_from is not None:
    raise ValueError("--restore_from is only valid with --start_stage > 1.")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  label = "smoke" if args.smoke_test else "8stage_400m"
  run_name = (
      args.run_name
      or (
          f"phase1_final_gait_agnostic_{label}_"
          f"seed_{args.seed}_{timestamp}"
      )
  )
  if run_name in {"", ".", ".."} or Path(run_name).name != run_name:
    raise ValueError("--run_name must be one plain directory name.")

  output_root = Path(args.output_root).expanduser().resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  run_dir = output_root / run_name
  try:
    run_dir.mkdir(exist_ok=False)
  except FileExistsError as exc:
    raise FileExistsError(f"Refusing to reuse run directory: {run_dir}") from exc

  restore_params = None
  if args.restore_from is not None:
    restore_path = Path(args.restore_from).expanduser().resolve()
    if not restore_path.is_file():
      raise FileNotFoundError(f"Missing restore params file: {restore_path}")
    restore_params = model.load_params(str(restore_path))
    if len(restore_params) != 3:
      raise ValueError(
          "The restore file must contain the (normalizer, policy, value) "
          "tuple written by a previous stage's final_policy."
      )

  stages_to_run = [
      stage
      for stage in CURRICULUM_STAGES
      if stage["stage"] >= args.start_stage
  ]
  curriculum_status_path = run_dir / "curriculum_status.json"
  curriculum_status = {
      "state": "running",
      "created_at": utc_now(),
      "updated_at": utc_now(),
      "run_directory": str(run_dir),
      "smoke_test": args.smoke_test,
      "start_stage": args.start_stage,
      "restored_from": args.restore_from,
      "stages": [
          {
              "stage": stage["stage"],
              "vx_range": list(stage["vx_range"]),
              "requested_num_timesteps": (
                  65_536 if args.smoke_test else stage["num_timesteps"]
              ),
              "expected_realized_num_timesteps": expected_realized_steps(
                  65_536 if args.smoke_test else stage["num_timesteps"],
                  2 if args.smoke_test else NUM_EVALS_PER_STAGE,
              ),
          }
          for stage in stages_to_run
      ],
      "expected_total_realized_num_timesteps": sum(
          expected_realized_steps(
              65_536 if args.smoke_test else stage["num_timesteps"],
              2 if args.smoke_test else NUM_EVALS_PER_STAGE,
          )
          for stage in stages_to_run
      ),
      "completed_stages": [],
      "current_stage": None,
      "current_stage_step": 0,
  }
  write_json(curriculum_status_path, curriculum_status)

  try:
    for stage in stages_to_run:
      restore_params, final_policy_path = run_stage(
          args,
          run_dir,
          stage,
          restore_params,
          curriculum_status,
          curriculum_status_path,
      )
      curriculum_status["completed_stages"].append(
          {
              "stage": stage["stage"],
              "vx_range": list(stage["vx_range"]),
              "final_policy": str(final_policy_path),
              "completed_at": utc_now(),
          }
      )
      curriculum_status.update(updated_at=utc_now())
      write_json(curriculum_status_path, curriculum_status)
    final_policy_path = run_dir / "final_policy"
    model.save_params(final_policy_path, restore_params)
    curriculum_status.update(
        state="completed",
        updated_at=utc_now(),
        final_policy=str(final_policy_path),
    )
    write_json(curriculum_status_path, curriculum_status)
    print(f"Curriculum complete. Artifacts: {run_dir}", flush=True)
  except BaseException as exc:
    curriculum_status.update(
        state="failed",
        updated_at=utc_now(),
        error=f"{type(exc).__name__}: {exc}",
    )
    write_json(curriculum_status_path, curriculum_status)
    raise


if __name__ == "__main__":
  main()
