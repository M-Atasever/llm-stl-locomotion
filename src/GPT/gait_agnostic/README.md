# GPT gait-agnostic pipeline

This is the final Phase 1 gait-agnostic implementation. It uses one shared STL formula across command regimes:

```text
G_[0,H](
  e_vx <= eps_vx and
  e_vy <= eps_vy and
  e_wz <= eps_wz and
  com_z >= z_min and
  max(abs(roll), abs(pitch)) <= theta_max and
  n_stance_contacts >= n_contact_min
)
```

The final configuration uses `H=1`. The unsupported upper height and upper contact-count predicates from the symbolic generation are disabled rather than assigned invented thresholds.

## Files

- `barkour.py`: MJX Barkour environment and signal/history extraction.
- `coeff_config.py`: final thresholds, predicate masks, horizon, and curriculum.
- `reward_config.py`: reward configuration.
- `stl_reward.py`: exact and differentiable STL robustness.
- `training_curriculum.py`: eight-stage, 400M-step curriculum trainer.
- `testing.py`: deterministic fixed-speed evaluator.
- `stl_specifications.json`: GPT-generated symbolic specification.

## Run

Set `BARKOUR_ROOT_PATH` as described in the repository README, then run from this directory:

```bash
python training_curriculum.py --smoke_test
python training_curriculum.py --seed 0
python testing.py --policy PLACEHOLDER --no_render
```

The checkpoint path is a `PLACEHOLDER` because trained weights are not included in the source repository.
