# Quadruped Locomotion with LLM-Generated Temporal Logic Specifications

This repository contains the Phase 1 final pipeline for training Barkour locomotion policies with dense rewards derived from LLM-generated Signal Temporal Logic (STL) specifications.

GPT and Qwen implementations are organized into two formulations:

- **Gait-agnostic:** one shared STL formula is used across commanded speeds, leaving the policy free to discover its locomotion pattern.
- **Multi-gait:** speed-conditioned STL objectives encode walk-like, trot-like, and bound-like behavior in one policy.

Only the Phase 1 pipeline source is included. Experiment sweeps, non-selected runs, checkpoints, logs, videos, and other ablation artifacts are intentionally excluded. The supplied Qwen source is included with unresolved settings explicitly marked `PLACEHOLDER`; see its README before running it.

## Repository layout

```text
llm-stl-locomotion/
├── README.md
├── mjx.yml
├── .gitignore
├── src/
│   ├── barkour.py                  # PLACEHOLDER shared entry point
│   ├── training.py                 # PLACEHOLDER shared entry point
│   ├── training_curriculum.py      # PLACEHOLDER shared entry point
│   ├── testing.py                  # PLACEHOLDER shared entry point
│   ├── GPT/
│   │   ├── gait_agnostic/           # final GPT gait-agnostic pipeline
│   │   └── multi_gait/              # final GPT multi-gait pipeline
│   └── Qwen/
│       ├── gait_agnostic/           # collection, q50 filter, curriculum
│       └── multi_gait/              # supplied two-stage training source
├── baselines/
│   ├── text2reward/
│   └── expert_oracle/
└── prompts/
    ├── gait_agnostic/               # GPT prompts plus Qwen/ snapshots
    └── multi_gait/                  # GPT prompts plus Qwen/ snapshots
```

The four top-level `src` modules are explicit placeholders because no single shared implementation was established across the models and formulations. Their source is kept separately under `src/GPT/` and `src/Qwen/` so the configurations are not silently mixed.

## Final GPT pipelines

| Formulation | Final source | STL setup | Training setup |
| --- | --- | --- | --- |
| Gait-agnostic | `src/GPT/gait_agnostic/` | Shared six-predicate formula; selected horizon `H=1` | Eight 50M-step curriculum stages, trained from scratch |
| Multi-gait | `src/GPT/multi_gait/` | Gait-conditioned walk/trot/bound predicates; selected horizon `H=20` | 400M-step continuation from the selected Phase 1 checkpoint |

Each directory contains its environment, STL configuration, reward implementation, trainer, evaluator, and generated symbolic specification. See the formulation-specific README before running it.

## Qwen pipelines

See [src/Qwen/README.md](src/Qwen/README.md) for the archive-derived pipeline, setup, and outstanding `PLACEHOLDER` items.

- **Gait-agnostic:** expert-trajectory collection, q50 (median) filtering, and an eight-stage, 400M-step command curriculum. The supplied filtered reward uses `H=1`; its change from the collection horizon has no supplied rationale.
- **Multi-gait:** a 400M-step walk/trot trainer and a separate 400M-step mixed-gait checkpoint-continuation trainer. The exact continuation checkpoint is `PLACEHOLDER`.
- **Unresolved multi-gait configuration:** the supplied buffer length `H=1` is below the eight-step gait-reward warmup and the declared per-mode windows. Numerical values are preserved; a setup guard stops execution until the intended final configuration is confirmed. It is not presented as a validated runnable final configuration.

## Environment setup

Create the pinned environment:

```bash
conda env create -f mjx.yml
conda activate mjx
```

Obtain the Barkour assets from MuJoCo Menagerie, then point the pipelines to the `google_barkour_vb` directory:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
export BARKOUR_ROOT_PATH="$PWD/third_party/mujoco_menagerie/google_barkour_vb"
```

## GPT quick checks

Gait-agnostic:

```bash
cd src/GPT/gait_agnostic
python training_curriculum.py --smoke_test
python testing.py --policy PLACEHOLDER --no_render
```

Multi-gait:

```bash
cd src/GPT/multi_gait
python training.py \
  --smoke_test \
  --resume_checkpoint PLACEHOLDER \
  --starting_checkpoint_steps 177930240
python testing.py --policy PLACEHOLDER --no-render
```

`PLACEHOLDER` denotes a checkpoint path that is not included in this source-only repository.

## Baselines

- `baselines/text2reward/` contains the gait-agnostic and multi-gait Text2Reward prompts and retained generated reward modules.
- `baselines/expert_oracle/` documents the expert-switching oracle. Its policy-download link remains a `PLACEHOLDER` until the authoritative Google Drive folder is provided.

## Papers and project media

- Previous paper: **PLACEHOLDER**
- Previous paper repository: **PLACEHOLDER**
- Current paper: **PLACEHOLDER**
- Project website and videos: **PLACEHOLDER**

## Citation

Previous paper citation:

```bibtex
PLACEHOLDER
```

Current paper citation:

```bibtex
PLACEHOLDER
```
