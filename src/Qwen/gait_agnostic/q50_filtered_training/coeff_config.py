"""Parameters for the q50-retained gait-agnostic reward and commands."""

from __future__ import annotations

import math
from typing import Dict

# Final training uses the supplied one-sample history at 50 Hz.
# Rationale for changing the collection horizon from 50 samples: PLACEHOLDER.
DT = 0.02
H = 1
H_WARMUP_MIN_VALID = 1
H_by_mode = H

# Active universal thresholds retained after q50 filtering.
eps_vx = 0.60
h_min = 0.18
ori_max = math.radians(10.0)

# Normalization scales.
velocity_margin_scale = 0.60
height_margin_scale = 0.03
orientation_margin_scale = math.radians(5.0)

ACTIVE_PREDICATES = (
    "vel_track_x",
    "safe_height",
    "safe_orientation",
)


def symbolic_thresholds() -> Dict[str, float]:
    return {
        "eps_vx": eps_vx,
        "h_min": h_min,
        "ori_max": ori_max,
    }


def validate_symbolic_thresholds() -> Dict[str, float]:
    values = symbolic_thresholds()
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Active STL thresholds must be finite.")
    if any(value < 0.0 for value in values.values()):
        raise ValueError("Active STL thresholds must be nonnegative.")
    return values


MODE_WALK = 0
MODE_TROT = 1
MODE_BOUND = 2

WALK_TO_TROT_ENTER = 0.72
TROT_TO_WALK_EXIT = 0.65
TROT_TO_BOUND_ENTER = 1.55
BOUND_TO_TROT_EXIT = 1.45

# Kept byte-for-value equivalent to the supplied environment/config.
regime_sample_probs = (0.25, 0.35, 0.40)
cmd_vx_range = (0.0, 1.9)
cmd_vy_range = (-0.2, 0.2)
cmd_yaw_range = (-0.2, 0.2)
bound_vx_sample_range = (1.45, 1.9)
