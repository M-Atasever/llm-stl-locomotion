# Text2Reward baseline

This directory contains separate prompts and retained generated reward modules for the two Phase 1 formulations:

- `gait_agnostic`: commanded velocity tracking without an explicit target gait.
- `gait_aware`: the multi-gait formulation with speed-dependent walk-like, trot-like, and bound-like objectives.

The `.txt` generation files preserve the model explanation and the `.py` files contain the corresponding reward modules. No additional generation candidates or baseline sweeps are included.
