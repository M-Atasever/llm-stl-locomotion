# Qwen: gait-agnostic pipeline

This directory contains the supplied expert-data collection, q50 selection,
retained reward runtime, and eight-stage command curriculum. The curriculum
imports the retained runtime directly; the separate walk/trot-only training
entrypoint and intermediate-stage diagnostic/repair evaluators are not included.

## Included stages

| Stage | Source | Supplied behavior |
| --- | --- | --- |
| Expert collection | `collection/collector.py` | Walk and trot; 50 trajectories of 500 steps per gait; no filtering during collection. |
| Component selection | `q50_filtering/compute_q50.py` | Pool walk/trot samples; retain a component only if its joint median is strictly positive. |
| Retained runtime | `q50_filtered_training/` | Forward-velocity tracking, height, and orientation predicates; `H=1`. |
| Curriculum training | `q50_curriculum_training/training_curriculum.py` | Eight sequential 50-million-step stages, 400 million steps in total. |
| Final evaluation | `q50_curriculum_training/testing.py` | 20 rollouts of 500 steps per command, 50-step speed warmup. |

The supplied selection JSON files retain `vel_track_x`, `safe_height`, and
`safe_orientation`, and exclude `low_slip`. They are selection artifacts, not
alternative training variants. Recomputing them requires the original expert
checkpoints and raw collection data, which are not included.

The collection reward uses a 50-sample history at `DT=0.02`; the supplied training
runtime uses a one-sample history at the same timestep. Both numerical settings
are preserved. Rationale for the horizon change: **PLACEHOLDER**. The generated
symbolic specification is not a direct statement of the final runtime's active
predicates.

The reward is gait-agnostic: its active predicates do not depend on the gait
mode. The supplied policy observation still includes the mode one-hot vector.
The curriculum samples forward velocity uniformly from zero to a stage maximum
of `0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9` m/s, preserving the supplied
lateral/yaw sampling in `[-0.2, 0.2]`. Its PPO network is four 128-unit layers;
seed is 0. `q50_filtered_training/training.py` contains only the shared settings
and helpers used by this curriculum, not another training variant.

## Required external inputs

Use the repository's environment setup. Set `BARKOUR_ROOT_PATH` to the directory
containing the Barkour `scene_mjx.xml` and its referenced model assets. This
archive did not include those assets or policy checkpoints.

- Barkour asset directory: **PLACEHOLDER**.
- Walk expert checkpoint: **PLACEHOLDER**.
- Trot expert checkpoint: **PLACEHOLDER**.
- Final curriculum checkpoint/download link: **PLACEHOLDER**.
- Raw collector generation record/request manifest: **PLACEHOLDER**.
- Evidence linking a specific trained checkpoint to this supplied source: **PLACEHOLDER**.

The collector retains the supplied expected expert-checkpoint hashes and rejects
a nonmatching checkpoint. The missing raw collector/request-manifest references
are marked `PLACEHOLDER`; no Qwen generation provenance is inferred from them.

## Commands

Run these from the repository root after replacing each `PLACEHOLDER` with the
corresponding real path. The model asset directory is read through an environment
variable; scripts do not change the caller's working directory.

```sh
export BARKOUR_ROOT_PATH="PLACEHOLDER"

python src/Qwen/gait_agnostic/collection/preflight.py \
  --walk-policy "PLACEHOLDER" --trot-policy "PLACEHOLDER"

python src/Qwen/gait_agnostic/collection/collector.py \
  --walk-policy "PLACEHOLDER" --trot-policy "PLACEHOLDER"

python src/Qwen/gait_agnostic/collection/validate_outputs.py \
  src/Qwen/gait_agnostic/collection/raw_data/walk
python src/Qwen/gait_agnostic/collection/validate_outputs.py \
  src/Qwen/gait_agnostic/collection/raw_data/trot

python src/Qwen/gait_agnostic/q50_filtering/compute_q50.py \
  --output-dir runs/qwen_gait_agnostic_filtering

python src/Qwen/gait_agnostic/q50_curriculum_training/training_curriculum.py \
  --output-root runs/qwen_gait_agnostic_curriculum

python src/Qwen/gait_agnostic/q50_curriculum_training/testing.py \
  --ckpt-path "PLACEHOLDER" --output-dir runs/qwen_gait_agnostic_evaluation
```

The collector defaults to ignored `collection/raw_data/`. New q50 analysis
defaults to ignored `runs/q50_filtering/` inside this directory and does not
overwrite the supplied selection JSON files. Analysis reports selection decisions;
it does not automatically rewrite the retained reward implementation.

Training and evaluation require new output directories. `--smoke` on the
curriculum is an operational check, not a reported experimental condition.
The final evaluator's commands are `0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 1.9, 2.0, 2.1`
m/s. Success requires surviving all 500 steps and post-warmup average forward
speed within ±15% of the command; cost of transport uses actuated-joint power,
robot mass, gravity, and planar distance as implemented in the supplied evaluator.
