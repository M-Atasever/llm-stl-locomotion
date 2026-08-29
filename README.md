# Quadruped Locomotion with LLM-Generated Temporal Logic Specifications

This repository contains the Phase 1 final pipeline for training Barkour locomotion policies with dense rewards derived from LLM-generated Signal Temporal Logic (STL) specifications.

Two GPT-based formulations are included:

- **Gait-agnostic:** one shared STL formula is used across commanded speeds, leaving the policy free to discover its locomotion pattern.
- **Multi-gait:** speed-conditioned STL objectives encode walk-like, trot-like, and bound-like behavior in one policy.

Only the selected Phase 1 implementations are included. Experiment sweeps, non-selected runs, checkpoints, logs, videos, and other ablation artifacts are intentionally excluded. Qwen files are not included in this version and will be added when they are available.

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
│   └── GPT/
│       ├── gait_agnostic/           # final GPT gait-agnostic pipeline
│       └── multi_gait/              # final GPT multi-gait pipeline
├── baselines/
│   ├── text2reward/
│   └── expert_oracle/
└── prompts/
    ├── gait_agnostic/
    └── multi_gait/
```

The four top-level `src` modules are explicit placeholders because no single shared implementation was established across the two final formulations. The complete runnable implementations are kept separately under `src/GPT/` so their verified configurations are not silently mixed.

## Final GPT pipelines

| Formulation | Final source | STL setup | Training setup |
| --- | --- | --- | --- |
| Gait-agnostic | `src/GPT/gait_agnostic/` | Shared six-predicate formula; selected horizon `H=1` | Eight 50M-step curriculum stages, trained from scratch |
| Multi-gait | `src/GPT/multi_gait/` | Gait-conditioned walk/trot/bound predicates; selected horizon `H=20` | 400M-step continuation from the selected Phase 1 checkpoint |

Each directory contains its environment, STL configuration, reward implementation, trainer, evaluator, and generated symbolic specification. See the formulation-specific README before running it.

## Environment setup

Create the pinned environment:

```bash
conda env create -f mjx.yml
conda activate mjx
```

Obtain the Barkour assets from MuJoCo Menagerie, then point both pipelines to the `google_barkour_vb` directory:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
export BARKOUR_ROOT_PATH="$PWD/third_party/mujoco_menagerie/google_barkour_vb"
```

## Quick checks

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
