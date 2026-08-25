# Quadruped Locomotion with LLM-Generated Temporal Logic Specifications


The full research project contains two formulations:

- **Gait-agnostic:** the policy tracks commanded `(vx, vy, yaw)` while remaining free to discover
  its own gait.
- **Multi-gait:** speed-dependent gait objectives encourage walk-like, trot-like, and bound-like
  locomotion.


## Repository layout

PLACEHOLDER


## Setup

Create the same Python environment used for MJX, then from the repository root:

```bash
python -m pip install -e .
```

### Barkour / MuJoCo Menagerie

The environment looks for MuJoCo Menagerie at:

```text
third_party/mujoco_menagerie/
```

## Quick visual policy check

Before the full benchmark:

```bash
python experiments/gait_agnostic/testing.py \
  --ckpt-path /path/to/checkpoint \
  --vx 1.3 --vy 0.0 --yaw 0.0 \
  --output-video policy_preview.mp4
```

## Full evaluation

```bash
python experiments/gait_agnostic/evaluate.py \
  --ckpt-path /path/to/checkpoint
```

The evaluator records tracking error, survival, success, CoT.

## Citation

Citation information will be added after the workshop paper metadata is finalized.
