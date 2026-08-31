"""Shared PPO settings and helpers used by the final command curriculum.

The archive's separate walk/trot-only training entrypoint is intentionally not
published here; the final training entrypoint is training_curriculum.py.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from flax.training import orbax_utils
from orbax import checkpoint as ocp

FULL_NUM_ENVS = 8192
FULL_BATCH_SIZE = 256
FULL_NUM_MINIBATCHES = 32
SEED = 0
NETWORK = (128, 128, 128, 128)


def _as_jsonable(value: Any) -> Any:
    value = jax.device_get(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_orbax_checkpoint(path, params):
    """Save PPO parameters as a loadable Orbax PyTree checkpoint directory."""
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(path, params, save_args=save_args)


def domain_randomize(sys, rng):
    """Apply the validated friction and actuator domain randomization."""

    @jax.vmap
    def rand(key):
        _, key = jax.random.split(key, 2)
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)

        _, key = jax.random.split(key, 2)
        parameter = jax.random.uniform(key, (1,), minval=-5, maxval=5)
        parameter = parameter + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(parameter)
        bias = sys.actuator_biasprm.at[:, 1].set(-parameter)
        return friction, gain, bias

    friction, gain, bias = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )
    randomized = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )
    return randomized, in_axes
