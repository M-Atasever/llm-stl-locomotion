# Quadruped Locomotion with LLM-Generated Temporal Logic Specifications

This repository contains the Phase 1 final pipeline for training Barkour locomotion policies with dense rewards derived from LLM-generated Signal Temporal Logic (STL) specifications.

GPT and Qwen implementations are organized into two formulations:

- **Gait-agnostic:** one shared STL formula is used across commanded speeds, leaving the policy free to discover its locomotion pattern.
- **Multi-gait:** speed-conditioned STL objectives encode walk-like, trot-like, and bound-like behavior in one policy.

## Repository layout

```text
llm-stl-locomotion/
├── README.md
├── mjx.yml
├── .gitignore
├── src/
│   ├── barkour.py                  
│   ├── training.py                 
│   ├── training_curriculum.py      
│   ├── testing.py                  
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

Each model's source is kept separately under `src/GPT/` and `src/Qwen/` so the configurations are not silently mixed.

## Final GPT pipelines

| Formulation | Final source | STL setup | Training setup |
| --- | --- | --- | --- |
| Gait-agnostic | `src/GPT/gait_agnostic/` | Shared six-predicate formula; selected horizon `H=1` | Eight 50M-step curriculum stages, trained from scratch |
| Multi-gait | `src/GPT/multi_gait/` | Gait-conditioned walk/trot/bound predicates; selected horizon `H=20` | 400M-step continuation from the selected Phase 1 checkpoint |

Each directory contains its environment, STL configuration, reward implementation, trainer, evaluator, and generated symbolic specification. See the formulation-specific README before running it.

## Qwen pipelines

| Formulation | Final source | STL setup | Training setup |
| --- | --- | --- | --- |
| Gait-agnostic | `src/Qwen/gait_agnostic/` | Shared q50-filtered reward; supplied horizon `H=1` | Expert-trajectory collection and q50 (median) filtering, followed by an eight-stage, 400M-step command curriculum |
| Multi-gait | `src/Qwen/multi_gait/` | Gait-conditioned walk/trot/bound reward | 400M-step walk/trot training followed by a separate 400M-step mixed-gait checkpoint continuation |

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

`PLACEHOLDER` denotes a checkpoint path.

## Baselines

- `baselines/text2reward/` contains the gait-agnostic and multi-gait Text2Reward prompts and retained generated reward modules.
- `baselines/expert_oracle/` documents the expert-switching oracle.

## Papers and project media

- [Previous paper](https://arxiv.org/abs/2607.00442)
- [Previous paper repository](https://github.com/M-Atasever/STL-based-Quadruped-Locomotion)
- Current paper: **PLACEHOLDER**
- [Project website and videos](https://stl-locomotion.github.io)

## Citation

Previous paper citation:

```bibtex
@misc{atasever2026learninggaitawarequadrupedlocomotion,
      title={Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications}, 
      author={Merve Atasever and Cagan Bakirci and Alfredo Reina Corona and Keyan Azbijari and Jyotirmoy V. Deshmukh},
      year={2026},
      eprint={2607.00442},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.00442}, 
}
```

Current paper citation:

```bibtex
PLACEHOLDER
```
