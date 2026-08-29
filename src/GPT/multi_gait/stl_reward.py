import os

import jax
import jax.numpy as jnp

from coeff_config import (
    ENABLE_BOUND_GAIT,
    H_by_mode,
    tau_max,
    gamma_tau,
    front_hind_phase_bound_target,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
)


SUPPORTED_SELECTION_QUANTILES = ("q50",)
SELECTION_QUANTILE = os.environ.get("STL_SELECTION_QUANTILE", "q50")
if SELECTION_QUANTILE not in SUPPORTED_SELECTION_QUANTILES:
    raise ValueError(
        "STL_SELECTION_QUANTILE must be one of "
        f"{SUPPORTED_SELECTION_QUANTILES}; got {SELECTION_QUANTILE!r}."
    )

BOUND_Q50_SELECTED_SPECIFICATIONS = (
    "bound__front_pair_synchronization",
    "bound__hind_pair_synchronization",
    "bound__front_pair_contact_fraction",
    "bound__hind_pair_contact_fraction",
    "bound__swing_clearance",
)

BOUND_Q50_EXCLUDED_SPECIFICATIONS = (
    "bound__front_hind_phase_relation",
    "bound__flight_fraction",
    "bound__duty_factor_upper_bound",
)

ACTIVE_SPECIFICATIONS_BY_SCOPE = {
    "shared": (
        "safety__bounded_roll",
        "safety__bounded_pitch",
        "safety__base_height_lower",
        "safety__bounded_joint_torque",
        "tracking__bounded_velocity_tracking_error",
    ),
    "walk": (
        "walk__min_contact_count",
        "walk__three_or_more_support_fraction",
        "walk__low_flight_fraction",
        "walk__high_duty_factor",
        "walk__swing_clearance",
    ),
    "trot": (
        "trot__diagonal_support_fraction",
        "trot__two_contact_fraction",
        "trot__diagonal_pair_synchronization",
        "trot__duty_factor_lower",
        "trot__swing_clearance",
    ),
    "bound": BOUND_Q50_SELECTED_SPECIFICATIONS,
}

EXCLUDED_SPECIFICATIONS_BY_QUANTILE = {
    "q50": (
        "safety__bounded_slip",
        "trot__duty_factor_upper",
        *BOUND_Q50_EXCLUDED_SPECIFICATIONS,
    ),
}

EXCLUDED_SPECIFICATIONS = EXCLUDED_SPECIFICATIONS_BY_QUANTILE[
    SELECTION_QUANTILE
]


_SHARED_WALK_TROT_REQUIRED_PARAMS = (
    "softmin_beta",
    "w_safe",
    "alpha_safe",
    "w_track",
    "alpha_track",
    "w_gait",
    "alpha_gait",

    # Shared safety.
    "theta_roll_max",
    "theta_pitch_max",
    "h_base_min",

    # Shared tracking.
    "e_vel_max",

    # Walk-like.
    "c_walk_min",
    "p_3plus_walk_min",
    "p_flight_walk_max",
    "duty_walk_min",
    "clearance_walk_min",

    # Trot-like.
    "p_diag2_trot_min",
    "p_2contact_trot_min",
    "eps_diag_sync_trot_max",
    "duty_trot_min",
    "clearance_trot_min",
)

_BOUND_REQUIRED_PARAMS = (
    # Bound-like; retained by the BOUND-expert q50 selection.
    "eps_front_sync_bound_max",
    "eps_hind_sync_bound_max",
    "p_front_pair_bound_min",
    "p_hind_pair_bound_min",
    "clearance_bound_min",
)

_REQUIRED_STL_PARAMS = _SHARED_WALK_TROT_REQUIRED_PARAMS + (
    _BOUND_REQUIRED_PARAMS if ENABLE_BOUND_GAIT else ()
)


def validate_stl_params(params):
    """Raises a clear error if fitted STL parameters have not been supplied."""
    missing = []
    for name in _REQUIRED_STL_PARAMS:
        if not hasattr(params, name) or getattr(params, name) is None:
            missing.append(name)
    if missing:
        raise ValueError(
            "STL_REWARD=True but the following STL parameters are unset: "
            f"{missing}. Fit these from expert trajectories and place the "
            "numerical values in reward_config.get_stl_config()."
        )


def _p(params, name):
    """Reads a scalar parameter as a JAX float."""
    value = getattr(params, name)
    if value is None:
        raise ValueError(f"STL parameter `{name}` is None.")
    return jnp.asarray(value, dtype=jnp.float32)


def tanh_norm(rho, alpha):
    return jnp.tanh(rho / jnp.maximum(alpha, 1e-6))


def compute_early_termination_penalty(done, magnitude):
    """Returns an unscaled cost only on physical-failure transitions."""
    return (
        -jnp.asarray(magnitude, dtype=jnp.float32)
        * jnp.asarray(done, dtype=jnp.float32)
    )


def _smooth_min(x, beta, axis=None):
    """Normalized smooth minimum.

    If all elements are equal to c, returns c.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    beta = jnp.maximum(jnp.asarray(beta, dtype=jnp.float32), 1e-6)

    if axis is None:
        n = x.size
    else:
        n = x.shape[axis]

    return -beta * (
        jax.nn.logsumexp(-x / beta, axis=axis) - jnp.log(jnp.maximum(n, 1))
    )


def _masked_smooth_min(x, mask, beta):
    """Smooth minimum over x where mask is true.

    Returns 0 if there are no valid entries.
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=bool)
    mask = jnp.broadcast_to(mask, x.shape)

    beta = jnp.maximum(jnp.asarray(beta, dtype=jnp.float32), 1e-6)
    count = jnp.sum(mask.astype(jnp.float32))
    has_any = count > 0.0

    z = jnp.where(mask, -x / beta, -jnp.inf)
    val = -beta * (jax.nn.logsumexp(z) - jnp.log(jnp.maximum(count, 1.0)))
    return jnp.where(has_any, val, 0.0)


def _masked_mean(x, mask):
    x = jnp.asarray(x, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=bool)
    mask = jnp.broadcast_to(mask, x.shape)
    m = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(m), 1.0)
    return jnp.sum(x * m) / denom


def _valid_mask_from_len(H, valid_len, horizon=None):
    """History convention: most recent sample is at index H-1.

    Leading rows are invalid until valid_len reaches H.
    """
    if horizon is None:
        horizon = H
    effective_len = jnp.minimum(valid_len, horizon)
    idx = jnp.arange(H)
    return idx >= (H - effective_len)


def _contact_summary(contact_history, valid_mask):
    """Computes allowed gait signals from the contact history.

    Foot order used by BarkourEnv:
      0: front-left  FL
      1: hind-left   HL
      2: front-right FR
      3: hind-right  HR
    """
    c = contact_history > 0.5

    FL = c[:, 0]
    HL = c[:, 1]
    FR = c[:, 2]
    HR = c[:, 3]

    contact_count = jnp.sum(c.astype(jnp.float32), axis=1)

    p_2contact = _masked_mean((contact_count == 2).astype(jnp.float32), valid_mask)
    p_3plus = _masked_mean((contact_count >= 3).astype(jnp.float32), valid_mask)
    p_flight = _masked_mean((contact_count == 0).astype(jnp.float32), valid_mask)

    diag_pair_1 = FL & HR
    diag_pair_2 = FR & HL
    exactly_two = contact_count == 2
    diag2 = exactly_two & (diag_pair_1 | diag_pair_2)
    # The attached coefficient's 0.99 threshold is diagonal purity among
    # exactly-two-contact states.  P_2contact separately constrains how much of
    # the complete window contains exactly two contacts.
    two_contact_mask = valid_mask & exactly_two
    two_contact_count = jnp.sum(two_contact_mask.astype(jnp.float32))
    p_diag2 = jnp.where(
        two_contact_count > 0.0,
        jnp.sum((two_contact_mask & diag2).astype(jnp.float32))
        / jnp.maximum(two_contact_count, 1.0),
        0.0,
    )

    front_pair_only = FL & FR & (~HL) & (~HR)
    hind_pair_only = HL & HR & (~FL) & (~FR)
    p_front_pair = _masked_mean(front_pair_only.astype(jnp.float32), valid_mask)
    p_hind_pair = _masked_mean(hind_pair_only.astype(jnp.float32), valid_mask)

    # Average foot duty factor over all feet.
    duty_factor = _masked_mean(c.astype(jnp.float32), valid_mask[:, None])

    # Synchronization-error proxies.
    #
    # These are mismatch fractions. They are zero if paired legs have identical
    # binary contact states over the evaluation window.
    diag_sync_error = 0.5 * (
        (FL != HR).astype(jnp.float32) + (FR != HL).astype(jnp.float32)
    )
    diag_sync_error = _masked_mean(diag_sync_error, valid_mask)

    front_sync_error = _masked_mean((FL != FR).astype(jnp.float32), valid_mask)
    hind_sync_error = _masked_mean((HL != HR).astype(jnp.float32), valid_mask)

    # Estimate hind-to-front phase by maximizing circular cross-correlation of
    # the front-only and hind-only pair signals.  The configured interval is
    # represented as a target midpoint plus a maximum absolute error.
    front_events = front_pair_only.astype(jnp.float32) * valid_mask
    hind_events = hind_pair_only.astype(jnp.float32) * valid_mask
    lag_scores = jnp.stack(
        [
            jnp.sum(jnp.roll(front_events, lag) * hind_events * valid_mask)
            for lag in range(contact_history.shape[0])
        ]
    )
    valid_count = jnp.sum(valid_mask.astype(jnp.float32))
    phase_fraction = jnp.argmax(lag_scores).astype(jnp.float32) / jnp.maximum(
        valid_count, 1.0
    )
    has_both_pairs = (jnp.sum(front_events) > 0.0) & (
        jnp.sum(hind_events) > 0.0
    )
    front_hind_phase_error = jnp.where(
        has_both_pairs,
        jnp.abs(phase_fraction - front_hind_phase_bound_target),
        1.0,
    )

    return dict(
        contact_count=contact_count,
        p_2contact=p_2contact,
        p_3plus=p_3plus,
        p_flight=p_flight,
        p_diag2=p_diag2,
        p_front_pair=p_front_pair,
        p_hind_pair=p_hind_pair,
        duty_factor=duty_factor,
        diag_sync_error=diag_sync_error,
        front_sync_error=front_sync_error,
        hind_sync_error=hind_sync_error,
        front_hind_phase_error=front_hind_phase_error,
    )


def _swing_clearance_rho(
    foot_clearance_history,
    contact_history,
    valid_mask,
    clearance_min,
    beta,
):
    """Robustness for Swing_clearance >= clearance_min.

    Evaluates only swing samples, defined by not-in-contact.
    If the window contains no swing samples, returns neutral robustness 0.
    """
    swing_mask = valid_mask[:, None] & (contact_history <= 0.5)
    margin = foot_clearance_history - clearance_min
    return _masked_smooth_min(margin, swing_mask, beta)


def _rho_shared_safety(reward_input, valid_mask, params):
    beta = _p(params, "softmin_beta")

    roll = reward_input["roll_history"]
    pitch = reward_input["pitch_history"]
    base_height = reward_input["base_height_history"]
    tau = reward_input["tau_history"]

    rho_roll = _masked_smooth_min(
        _p(params, "theta_roll_max") - jnp.abs(roll),
        valid_mask,
        beta,
    )
    rho_pitch = _masked_smooth_min(
        _p(params, "theta_pitch_max") - jnp.abs(pitch),
        valid_mask,
        beta,
    )
    rho_height_low = _masked_smooth_min(
        base_height - _p(params, "h_base_min"),
        valid_mask,
        beta,
    )
    # STL predicate Joint_torque <= tau_max.
    # tau_max is currently a known robot limit in coeff_config.py.
    rho_torque = _masked_smooth_min(
        tau_max - jnp.abs(tau),
        valid_mask[:, None],
        beta,
    )

    return _smooth_min(
        jnp.array(
            [
                rho_roll,
                rho_pitch,
                rho_height_low,
                rho_torque,
            ],
            dtype=jnp.float32,
        ),
        beta,
    ), rho_torque


def _rho_shared_tracking(reward_input, valid_mask, params):
    beta = _p(params, "softmin_beta")
    vel_err = reward_input["velocity_tracking_error_history"]
    return _masked_smooth_min(
        _p(params, "e_vel_max") - vel_err,
        valid_mask,
        beta,
    )


def _rho_walk_gait(reward_input, valid_mask, params):
    beta = _p(params, "softmin_beta")

    contact_history = reward_input["contact_history"]
    foot_clearance_history = reward_input["foot_clearance_history"]

    s = _contact_summary(contact_history, valid_mask)

    rho_contact_count = _masked_smooth_min(
        s["contact_count"] - _p(params, "c_walk_min"),
        valid_mask,
        beta,
    )
    rho_p_3plus = s["p_3plus"] - _p(params, "p_3plus_walk_min")
    rho_p_flight = _p(params, "p_flight_walk_max") - s["p_flight"]
    rho_duty = s["duty_factor"] - _p(params, "duty_walk_min")
    rho_clearance = _swing_clearance_rho(
        foot_clearance_history,
        contact_history,
        valid_mask,
        _p(params, "clearance_walk_min"),
        beta,
    )

    return _smooth_min(
        jnp.array(
            [
                rho_contact_count,
                rho_p_3plus,
                rho_p_flight,
                rho_duty,
                rho_clearance,
            ],
            dtype=jnp.float32,
        ),
        beta,
    )


def _rho_trot_gait(reward_input, valid_mask, params):
    beta = _p(params, "softmin_beta")

    contact_history = reward_input["contact_history"]
    foot_clearance_history = reward_input["foot_clearance_history"]

    s = _contact_summary(contact_history, valid_mask)

    rho_p_diag2 = s["p_diag2"] - _p(params, "p_diag2_trot_min")
    rho_p_2contact = s["p_2contact"] - _p(params, "p_2contact_trot_min")
    rho_diag_sync = (
        _p(params, "eps_diag_sync_trot_max") - s["diag_sync_error"]
    )
    rho_duty_low = s["duty_factor"] - _p(params, "duty_trot_min")
    rho_clearance = _swing_clearance_rho(
        foot_clearance_history,
        contact_history,
        valid_mask,
        _p(params, "clearance_trot_min"),
        beta,
    )

    return _smooth_min(
        jnp.array(
            [
                rho_p_diag2,
                rho_p_2contact,
                rho_diag_sync,
                rho_duty_low,
                rho_clearance,
            ],
            dtype=jnp.float32,
        ),
        beta,
    )


def _rho_bound_gait(reward_input, valid_mask, params):
    beta = _p(params, "softmin_beta")

    contact_history = reward_input["contact_history"]
    foot_clearance_history = reward_input["foot_clearance_history"]

    s = _contact_summary(contact_history, valid_mask)

    rho_front_sync = (
        _p(params, "eps_front_sync_bound_max") - s["front_sync_error"]
    )
    rho_hind_sync = (
        _p(params, "eps_hind_sync_bound_max") - s["hind_sync_error"]
    )
    rho_p_front_pair = (
        s["p_front_pair"] - _p(params, "p_front_pair_bound_min")
    )
    rho_p_hind_pair = (
        s["p_hind_pair"] - _p(params, "p_hind_pair_bound_min")
    )
    rho_clearance = _swing_clearance_rho(
        foot_clearance_history,
        contact_history,
        valid_mask,
        _p(params, "clearance_bound_min"),
        beta,
    )

    return _smooth_min(
        jnp.array(
            [
                rho_front_sync,
                rho_hind_sync,
                rho_p_front_pair,
                rho_p_hind_pair,
                rho_clearance,
            ],
            dtype=jnp.float32,
        ),
        beta,
    )


def _rho_mode_gait(reward_input, mode, valid_mask, params):
    """Mode-conditioned gait STL robustness."""
    branches = [
        lambda _: _rho_walk_gait(reward_input, valid_mask, params),
        lambda _: _rho_trot_gait(reward_input, valid_mask, params),
        lambda _: _rho_bound_gait(reward_input, valid_mask, params),
    ]
    maximum_mode = MODE_BOUND if ENABLE_BOUND_GAIT else MODE_TROT
    mode = jnp.clip(mode, MODE_WALK, maximum_mode)
    return jax.lax.switch(mode, branches, operand=None)


def _masked_tau_effort(tau, valid_mask):
    tau_sq = jnp.sum(jnp.square(tau), axis=1)
    return _masked_mean(tau_sq, valid_mask)


def _masked_action_rate(action_delta_history, valid_mask):
    return _masked_mean(action_delta_history, valid_mask)


def _group_params(params, weights_override):
    """Returns reward group params.

    Expected order:
      [w_safe, alpha_safe, w_track, alpha_track, w_gait, alpha_gait]

    For backward compatibility, a length-2 override array is interpreted as:
      [w_safe, alpha_safe]
    """
    base = jnp.array(
        [
            _p(params, "w_safe"),
            _p(params, "alpha_safe"),
            _p(params, "w_track"),
            _p(params, "alpha_track"),
            _p(params, "w_gait"),
            _p(params, "alpha_gait"),
        ],
        dtype=jnp.float32,
    )

    if weights_override is None:
        return base

    if weights_override.shape[0] == 2:
        return base.at[0].set(weights_override[0]).at[1].set(weights_override[1])

    if weights_override.shape[0] == 6:
        return weights_override

    raise ValueError(
        "weights_override must have shape (2,) or (6,). "
        "Use [w_safe, alpha_safe] or "
        "[w_safe, alpha_safe, w_track, alpha_track, w_gait, alpha_gait]."
    )


def reward_step(
    reward_input,
    commands,
    mode,
    valid_len,
    stl_params,
    weights_override=None,
):
    """Computes one STL-shaped reward from the finite history window.

    Args:
      reward_input: dict of histories, each with leading invalid rows and most
        recent value at index H-1.
      commands: current command, kept for API compatibility.
      mode: current gait mode.
      valid_len: number of valid history rows, clipped to H.
      stl_params: reward_config.stl ConfigDict with fitted numerical params.
      weights_override: optional JAX array of reward group weights.

    Returns:
      tuple used by BarkourEnv.step.
    """
    del commands

    tau = reward_input["tau_history"]
    H = tau.shape[0]
    valid_len = jnp.clip(valid_len, 0, H)
    maximum_mode = MODE_BOUND if ENABLE_BOUND_GAIT else MODE_TROT
    mode = jnp.clip(mode, MODE_WALK, maximum_mode)
    mode_horizon = jnp.asarray(H_by_mode, dtype=jnp.int32)[mode]
    valid_mask = _valid_mask_from_len(H, valid_len, mode_horizon)

    (
        w_safe,
        alpha_safe,
        w_track,
        alpha_track,
        w_gait,
        alpha_gait,
    ) = _group_params(stl_params, weights_override)

    rho_safety_raw, rho_torque_raw = _rho_shared_safety(
        reward_input,
        valid_mask,
        stl_params,
    )
    rho_tracking_raw = _rho_shared_tracking(
        reward_input,
        valid_mask,
        stl_params,
    )
    rho_gait_raw = _rho_mode_gait(
        reward_input,
        mode,
        valid_mask,
        stl_params,
    )

    rho_safety = tanh_norm(rho_safety_raw, alpha_safe)
    rho_tracking = tanh_norm(rho_tracking_raw, alpha_track)
    rho_gait = tanh_norm(rho_gait_raw, alpha_gait)

    tau_effort = _masked_tau_effort(tau, valid_mask)
    action_rate_effort = _masked_action_rate(
        reward_input["action_delta_history"],
        valid_mask,
    )

    reward = (
        w_safe * rho_safety
        + w_track * rho_tracking
        + w_gait * rho_gait
        - gamma_tau * tau_effort
    )

    return (
        reward,
        tau_effort,
        action_rate_effort,
        rho_safety,
        rho_tracking,
        rho_gait,
        rho_safety_raw,
        rho_tracking_raw,
        rho_gait_raw,
        rho_torque_raw,
    )
