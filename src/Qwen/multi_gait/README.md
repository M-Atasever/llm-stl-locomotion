# Qwen: multi-gait pipeline

This directory contains the multi-gait code supplied in `qwen.zip`, organized
without changing the numerical reward or PPO settings. The original
`multi_gait_training/src` and `multi_gait_training/configs` split is retained.

## Required confirmation: PLACEHOLDER

The intended final history configuration is **PLACEHOLDER**. The supplied
`configs/coeff_config.py` sets `H = 1`, `H_by_mode = (30, 24, 24)`, and
`H_WARMUP_MIN_VALID = 8`. `BarkourEnv` allocates only `H` history entries and caps
the valid history length at `H`. The reward enables gait-pattern terms only
after eight valid entries, so those terms never activate with the supplied
configuration. The collector also requires eight entries and would write NaN
robustness scores throughout.

No replacement value was inferred. A validation check now raises an explicit
`PLACEHOLDER` error before model loading or training/collection output creation.
Confirm the final settings before running this pipeline. CLI `--help` and
inspection of the source remain available.

## Included stages

1. `generated/qwen_stl_specs.json` is the supplied symbolic specification output.
   The reward implementation contains the numerical, filtered predicates; it
   does not load this JSON at runtime.
2. `collection/collect_stl_data.py` collects walking and trotting expert
   trajectories and robustness diagnostics. Its byte-identical duplicate,
   `qwen_generated_collection_script.py`, is omitted. Expert checkpoint paths
   are **PLACEHOLDER** and must be supplied explicitly. The collector supports
   both supplied observation layouts: 510 features with a gait-mode flag and
   legacy 465-feature expert checkpoints.
3. `generated/component_q50_selection.json` is the supplied frozen walk/trot
   component-selection record. It protects forward-velocity tracking, retains
   trotting slip, and records excluded components. The referenced statistics
   file and the multi-gait quantile-calculation script were not supplied:
   **PLACEHOLDER**. This record contains no bounding-expert selection results;
   those results are **PLACEHOLDER**, not inferred from walk/trot data.
4. `multi_gait_training/src/training.py` trains walk/trot commands for
   400,000,000 timesteps by default.
5. `multi_gait_training/src/training_bound.py` explicitly continues a supplied
   checkpoint for an additional 400,000,000 timesteps by default, with mixed
   walk/trot/bound command sampling. These are sequential stages, not two
   alternative reward ablations. The exact final first-stage checkpoint and
   final trained policy are **PLACEHOLDER**; no checkpoint files were supplied.
6. `multi_gait_training/src/test_updated.py` computes transportation cost,
   survival, and velocity-tracking success. `visual_test.py` separately renders
   videos for the supplied commands 0.5, 1.0, and 1.9 m/s. Neither is a training
   variant.

The implemented reward is the weighted sum of bounded safety, forward-velocity
tracking, and gait-pattern terms, minus torque effort. Height and several gait
scores remain diagnostics after filtering. Timing scores are computed but are
not added to the final reward in the supplied implementation. With the
unresolved `H = 1`, the gait-pattern contribution is also zero; this is why the
setup check blocks execution.

## Setup and entrypoints

Use the repository's [`mjx.yml`](../../../mjx.yml) environment instructions.
Run the commands below from the repository root after resolving the history
configuration. Keep experiment outputs under the ignored `runs/` directory.

The model assets are external. By default the environment expects
`mujoco_menagerie/google_barkour_vb/scene_mjx.xml` relative to the current working
directory. To use another existing asset directory, set:

```sh
export BARKOUR_ROOT_PATH=PLACEHOLDER
```

`BARKOUR_ROOT_PATH` must name the directory containing `scene_mjx.xml` and its
referenced assets. `MUJOCO_GL` is respected if already set; otherwise it defaults
to `glfw` on macOS and `egl` elsewhere.

```sh
python src/Qwen/multi_gait/multi_gait_training/src/preflight.py

python src/Qwen/multi_gait/collection/collect_stl_data.py \
  --walk-ckpt PLACEHOLDER --trot-ckpt PLACEHOLDER \
  --output-dir runs/qwen_multi_gait/collection

python src/Qwen/multi_gait/multi_gait_training/src/training.py \
  --output-dir runs/qwen_multi_gait/walk_trot

python src/Qwen/multi_gait/multi_gait_training/src/training_bound.py \
  --restore-checkpoint PLACEHOLDER \
  --output-dir runs/qwen_multi_gait/mixed_continuation

python src/Qwen/multi_gait/multi_gait_training/src/test_updated.py \
  --ckpt-path PLACEHOLDER \
  --rollouts-csv runs/qwen_multi_gait/evaluation/rollouts.csv \
  --summary-csv runs/qwen_multi_gait/evaluation/summary.csv \
  --summary-json runs/qwen_multi_gait/evaluation/summary.json

python src/Qwen/multi_gait/multi_gait_training/src/visual_test.py \
  --ckpt-path PLACEHOLDER --output-dir runs/qwen_multi_gait/videos
```

The trainers accept `--smoke` for a short operational PPO check; it is not a
separate experiment and does not bypass the unresolved configuration check.
Evaluation defaults to 20 rollouts per speed, 500 steps, and 50 warmup steps.
Success requires survival and post-warmup mean forward velocity within 15% of
the command. CoT uses absolute mechanical work divided by mass, gravity, and
accumulated planar travel distance. The video utility retains the supplied
behavior of continuing after termination and flags terminated runs in its
summary; its averages are not the quantitative success metric.

## Integration changes

Only repository packaging and operational checks were added: local import
paths, configurable model-asset location, non-overriding platform-aware rendering
defaults, and the fail-closed history check. Numerical configuration, reward
formulas, training stages, and evaluation definitions were retained. The
supplied JSON files remain provenance artifacts, not fabricated run results.

No expert trajectories, trained weights, evaluation outputs, duplicate
collection script, or alternate reward experiments are included. Full training
and checkpoint evaluation have not been performed as part of this import.
