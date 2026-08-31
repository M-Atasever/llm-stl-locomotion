"""JAX-traceable q50-retained gait-agnostic STL reward."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax
import jax.numpy as jnp

try:
    from . import coeff_config as _coeff
except ImportError:
    import coeff_config as _coeff

ACTIVE_PREDICATES = _coeff.ACTIVE_PREDICATES
H = _coeff.H
eps_vx = _coeff.eps_vx
h_min = _coeff.h_min
height_margin_scale = _coeff.height_margin_scale
orientation_margin_scale = _coeff.orientation_margin_scale
ori_max = _coeff.ori_max
velocity_margin_scale = _coeff.velocity_margin_scale

ACTIVE_SIGNAL_KEYS = (
    "velocity_error_x",
    "base_height",
    "orientation_error",
)
DIAGNOSTIC_SIGNAL_KEYS = ("velocity_error_y", "velocity_error_yaw")
SIGNAL_KEYS = ACTIVE_SIGNAL_KEYS + DIAGNOSTIC_SIGNAL_KEYS
METRIC_KEYS = (
    *ACTIVE_PREDICATES,
    "rho_task_safety",
    "diagnostic_abs_velocity_error_y",
    "diagnostic_abs_velocity_error_yaw",
    "total_stl_reward",
)


def _validate_reward_input(reward_input: Mapping[str, jax.Array]) -> None:
    missing = [key for key in SIGNAL_KEYS if key not in reward_input]
    if missing:
        raise KeyError("Missing STL signal histories: " + ", ".join(missing))
    bad_shapes = [
        f"{key}={tuple(reward_input[key].shape)}"
        for key in SIGNAL_KEYS
        if reward_input[key].ndim != 1 or reward_input[key].shape[0] != H
    ]
    if bad_shapes:
        raise ValueError(
            f"Every required signal history must have shape ({H},): "
            + ", ".join(bad_shapes)
        )


def _valid_mask(valid_len: jax.Array) -> jax.Array:
    valid_len = jnp.clip(jnp.asarray(valid_len, dtype=jnp.int32), 0, H)
    return jnp.arange(H, dtype=jnp.int32) >= H - valid_len


def _always(robustness: jax.Array, mask: jax.Array) -> jax.Array:
    result = jnp.min(jnp.where(mask, robustness, jnp.inf))
    return jnp.where(jnp.any(mask), result, jnp.asarray(0.0, robustness.dtype))


def _diagnostic_max_abs(signal: jax.Array, mask: jax.Array) -> jax.Array:
    result = jnp.max(jnp.where(mask, jnp.abs(signal), -jnp.inf))
    return jnp.where(jnp.any(mask), result, jnp.asarray(0.0, signal.dtype))


def reward_step(
    reward_input: Mapping[str, jax.Array],
    commands: jax.Array,
    mode: jax.Array,
    valid_len: jax.Array,
    weights_override=None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Evaluate G_[0,H] over the q50-retained normalized predicates."""
    del commands, mode
    if weights_override is not None:
        raise ValueError("The retained quantitative conjunction has no weights.")
    _validate_reward_input(reward_input)
    mask = _valid_mask(valid_len)

    metrics = {
        "vel_track_x": _always(
            (eps_vx - jnp.abs(reward_input["velocity_error_x"]))
            / velocity_margin_scale,
            mask,
        ),
        "safe_height": _always(
            (reward_input["base_height"] - h_min) / height_margin_scale,
            mask,
        ),
        "safe_orientation": _always(
            (ori_max - jnp.abs(reward_input["orientation_error"]))
            / orientation_margin_scale,
            mask,
        ),
    }
    metrics["rho_task_safety"] = jnp.min(
        jnp.stack(
            (
                metrics["vel_track_x"],
                metrics["safe_height"],
                metrics["safe_orientation"],
            )
        )
    )
    metrics["diagnostic_abs_velocity_error_y"] = _diagnostic_max_abs(
        reward_input["velocity_error_y"], mask
    )
    metrics["diagnostic_abs_velocity_error_yaw"] = _diagnostic_max_abs(
        reward_input["velocity_error_yaw"], mask
    )
    metrics["total_stl_reward"] = metrics["rho_task_safety"]
    reward = metrics["total_stl_reward"]
    return reward, metrics
