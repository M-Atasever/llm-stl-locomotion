"""Final command-relative velocity error shared by all source paths."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from coeff_config import (
    track_axis_weights,
    velocity_command_floor,
    velocity_error_scales,
)


class VelocityTrackingError(NamedTuple):
  """Normalized per-axis errors and the final weighted scalar."""

  forward_normalized_error: jnp.ndarray
  lateral_normalized_error: jnp.ndarray
  yaw_normalized_error: jnp.ndarray
  velocity_error: jnp.ndarray


def command_relative_weighted_tracking_error(
    command,
    body_linear_velocity,
    body_yaw_rate,
) -> VelocityTrackingError:
  """Computes the final tracking treatment in the body frame."""
  command = jnp.asarray(command)
  body_linear_velocity = jnp.asarray(body_linear_velocity)
  body_yaw_rate = jnp.asarray(body_yaw_rate)

  dtype = jnp.result_type(
      command.dtype,
      body_linear_velocity.dtype,
      body_yaw_rate.dtype,
      jnp.float32,
  )
  command = command.astype(dtype)
  body_linear_velocity = body_linear_velocity.astype(dtype)
  body_yaw_rate = body_yaw_rate.astype(dtype)

  fixed_scales = jnp.asarray(velocity_error_scales, dtype=dtype)
  forward_denominator = jnp.maximum(
      jnp.abs(command[..., 0]),
      jnp.asarray(velocity_command_floor, dtype=dtype),
  )
  forward_error = (
      jnp.abs(command[..., 0] - body_linear_velocity[..., 0])
      / forward_denominator
  )
  lateral_error = (
      jnp.abs(command[..., 1] - body_linear_velocity[..., 1])
      / fixed_scales[1]
  )
  yaw_error = jnp.abs(command[..., 2] - body_yaw_rate) / fixed_scales[2]

  weights = jnp.asarray(track_axis_weights, dtype=dtype)
  velocity_error = (
      weights[0] * forward_error
      + weights[1] * lateral_error
      + weights[2] * yaw_error
  )
  return VelocityTrackingError(
      forward_normalized_error=forward_error,
      lateral_normalized_error=lateral_error,
      yaw_normalized_error=yaw_error,
      velocity_error=velocity_error,
  )


def instantaneous_tracking_robustness(velocity_error):
  """Returns the unchanged atomic predicate robustness ``1 - error``."""
  velocity_error = jnp.asarray(velocity_error)
  return jnp.asarray(1.0, dtype=velocity_error.dtype) - velocity_error
