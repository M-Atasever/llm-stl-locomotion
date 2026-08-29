# GPT Phase 1 pipelines

This directory contains the two final GPT-based Phase 1 implementations.

## Gait-agnostic

`gait_agnostic/` applies one shared six-predicate STL formula to walk, trot, and bound command regimes. The active runtime predicates cover forward/lateral/yaw tracking, center-of-mass height, body tilt, and contact count. The selected final temporal horizon is `H=1`.

## Multi-gait

`multi_gait/` conditions the STL reward on walk-like, trot-like, and bound-like modes. It retains the selected gait-specific predicates and uses the final temporal horizon `H=20`.

The directories intentionally contain only runnable final source, configuration, evaluation code, and the generated symbolic specification. Model checkpoints and experiment outputs are not included.
