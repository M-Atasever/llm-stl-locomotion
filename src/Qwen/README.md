# Qwen Phase 1 source

This directory contains the Phase 1 pipeline source selected from the supplied `qwen.zip`. The archive's model-call scripts name `qwen3.6-plus`; complete response/request provenance and checkpoint identities were not supplied, so their verification is `PLACEHOLDER`.

## Formulations

| Formulation | Included pipeline | Status |
| --- | --- | --- |
| [Gait-agnostic](gait_agnostic/README.md) | Expert collection → q50 filtering → shared filtered reward → eight-stage command curriculum → evaluation | Supplied curriculum requests 8 × 50M steps; model assets and policies are external. |
| [Multi-gait](multi_gait/README.md) | Expert collection and retained component-q50 selection → walk/trot training → mixed-gait continuation → evaluation | Supplied temporal settings conflict; execution is guarded pending confirmation. |

Gait-agnostic uses one shared formula across commanded speeds. Multi-gait conditions its reward on walk/trot/bound modes. These are model-specific source trees, not interchangeable versions of the GPT implementation.

The Qwen multi-gait archive sets the history buffer to `H=1`, the per-mode windows to `(30, 24, 24)`, and gait warmup to `8`. With the supplied buffer, gait rewards cannot activate. These numerical values have not been changed: the intended final configuration is **PLACEHOLDER**. A validation guard reports the issue before training or environment construction.

## Setup

Use the repository's `mjx.yml` environment and set `BARKOUR_ROOT_PATH` to the absolute `google_barkour_vb` asset directory, as described in the root README. The pinned environment is CUDA/Linux-oriented; a full Qwen training run has not been reproduced by this repository integration. See each formulation's README for entry points and required external inputs.

No checkpoint, expert-policy binary, raw trajectory dump, or trained-policy result is included. Policy locations, exact checkpoint identities, and any missing selection provenance remain **PLACEHOLDER**.

## What was retained

- Supplied environments, reward implementations, active configuration, required training stages, collection/filtering code, and evaluators.
- Generated symbolic specifications and small q50 selection records that document the filtering stage; these are not performance-result claims.
- The six supplied prompt snapshots under `prompts/gait_agnostic/Qwen/` and `prompts/multi_gait/Qwen/`.

Duplicate collection code, intermediate/repair evaluation scripts, the separate gait-agnostic walk/trot-only training entry point, and API replay/re-prompt scripts are excluded. The combined prompt module is also excluded because it contains Phase 2 content. Required shared training helpers are retained for the gait-agnostic curriculum.

Packaging changes are limited to portable paths/imports, clear missing-input errors, and the unresolved-configuration guard. Scientific reward thresholds and training schedules are preserved rather than inferred or retuned. Prompt snapshots are reference inputs, not instructions to run additional experiments.
