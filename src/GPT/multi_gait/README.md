# GPT multi-gait pipeline

This is the final Phase 1 multi-gait implementation. Its STL reward is conditioned on speed-dependent walk-like, trot-like, and bound-like modes. The final configuration uses `H=20` and the selected q50 predicate set.

## Files

- `barkour.py`: MJX Barkour environment, gait selection, and signal histories.
- `coeff_config.py`: final thresholds, gait-specific predicates, and horizon.
- `reward_config.py`: reward configuration.
- `stl_reward.py`: exact and differentiable gait-conditioned STL robustness.
- `velocity_tracking.py`: command-relative tracking treatment used by training and evaluation.
- `training.py`: 400M-step continuation trainer.
- `testing.py`: deterministic fixed-speed evaluator.
- `stl_specifications.json`: GPT-generated symbolic multi-gait specification.

## Run

Set `BARKOUR_ROOT_PATH` as described in the repository README. This trainer is intentionally warm-start-only, so an authoritative Phase 1 checkpoint must be supplied:

```bash
python training.py \
  --smoke_test \
  --resume_checkpoint PLACEHOLDER \
  --starting_checkpoint_steps 177930240

python training.py \
  --resume_checkpoint PLACEHOLDER \
  --starting_checkpoint_steps 177930240 \
  --seed 0

python testing.py --policy PLACEHOLDER --no-render
```

The checkpoint and trained-policy paths remain `PLACEHOLDER` values because model weights are not included in this source repository.
