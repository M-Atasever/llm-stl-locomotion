# Qwen Phase 1 source

## Formulations

| Formulation | Included pipeline | Status |
| --- | --- | --- |
| [Gait-agnostic](gait_agnostic/README.md) | Expert collection → q50 filtering → shared filtered reward → eight-stage command curriculum → evaluation | Supplied curriculum requests 8 × 50M steps; model assets and policies are external. |
| [Multi-gait](multi_gait/README.md) | Expert collection and retained component-q50 selection → walk/trot training → mixed-gait continuation → evaluation | Supplied temporal settings conflict; execution is guarded pending confirmation. |

Gait-agnostic uses one shared formula across commanded speeds. Multi-gait conditions its reward on walk/trot/bound modes. 

The Qwen multi-gait archive sets the history buffer to `H=1`, the per-mode windows to `(30, 24, 24)`, and gait warmup to `8`. With the supplied buffer, gait rewards cannot activate. 

## Setup

Use the repository's `mjx.yml` environment and set `BARKOUR_ROOT_PATH` to the absolute `google_barkour_vb` asset directory, as described in the root README. The pinned environment is CUDA/Linux-oriented. See each formulation's README for entry points and required external inputs.

## What was retained

- Supplied environments, reward implementations, active configuration, required training stages, collection/filtering code, and evaluators.
- Generated symbolic specifications and small q50 selection records that document the filtering stage; these are not performance-result claims.
- The six supplied prompt snapshots under `prompts/gait_agnostic/Qwen/` and `prompts/multi_gait/Qwen/`.
