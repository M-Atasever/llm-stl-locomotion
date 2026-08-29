import jax
import jax.numpy as jnp

from coeff_config import (
    BOUND_ACTIVE_PREDICATE_MASK,
    H_WARMUP_MIN_VALID,
    MODE_TROT,
    MODE_WALK,
    PREDICATE_REWARD_WEIGHTS,
    TROT_ACTIVE_PREDICATE_MASK,
    WALK_ACTIVE_PREDICATE_MASK,
    beta,
    com_z_margin_scale,
    eps_vx,
    eps_vy,
    eps_wz,
    gamma_tau,
    min_contacts_margin_scale,
    n_contact_min,
    tau_max,
    theta_max,
    tilt_margin_scale,
    velocity_vx_margin_scale,
    velocity_vy_margin_scale,
    velocity_wz_margin_scale,
    z_min,
)


_EPS = 1e-6

if len(PREDICATE_REWARD_WEIGHTS) != 6:
    raise ValueError("PREDICATE_REWARD_WEIGHTS must follow the six-predicate order.")
if any(weight <= 0.0 for weight in PREDICATE_REWARD_WEIGHTS):
    raise ValueError("Every predicate reward weight must be strictly positive.")


def _valid_suffix_mask(length, valid_len):
    """Returns a mask selecting the valid right-aligned history suffix."""
    valid_len = jnp.clip(valid_len, H_WARMUP_MIN_VALID, length)
    return jnp.arange(length) >= (length - valid_len)


def _masked_min_last(values, valid_len):
    """Exact minimum over the valid right-aligned history suffix."""
    mask = _valid_suffix_mask(values.shape[0], valid_len)
    return jnp.min(jnp.where(mask, values, jnp.inf))


def _masked_mean_last(values, valid_len):
    """Mean over the valid right-aligned history suffix."""
    mask = _valid_suffix_mask(values.shape[0], valid_len)
    mask_f = mask.astype(values.dtype)
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(
        jnp.sum(mask_f), 1.0
    )


def _smooth_min(values, sharpness, axis, mask=None, base_weights=None):
    """Numerically stable, idempotent weighted smooth minimum.

    Increasing sharpness approaches the exact minimum. Equal inputs aggregate
    to that same value, avoiding a predicate-count-dependent offset. Optional
    positive base_weights act as prior importance weights inside the softmax;
    they do not alter exact STL robustness or predicate satisfaction.
    """
    values = jnp.asarray(values, dtype=jnp.float32)
    sharpness = jnp.maximum(jnp.asarray(sharpness, dtype=values.dtype), _EPS)
    logits = -sharpness * values

    if base_weights is not None:
        base_weights = jnp.asarray(base_weights, dtype=values.dtype)
        logits = logits + jnp.log(jnp.maximum(base_weights, _EPS))

    if mask is None:
        safe_values = values
    else:
        mask = jnp.asarray(mask, dtype=bool)
        while mask.ndim < values.ndim:
            mask = jnp.expand_dims(mask, axis=-1)
        logits = jnp.where(mask, logits, -jnp.inf)
        safe_values = jnp.where(mask, values, 0.0)

    weights = jax.nn.softmax(logits, axis=axis)
    return jnp.sum(weights * safe_values, axis=axis)


def reward_step(reward_input, commands, mode, valid_len, weights_override=None):
    """Computes exact and smooth robustness for the active STL formula.

    stl_history has shape (H, 6) with columns:
      0: e_vx = abs(v_x - v_x_cmd)
      1: e_vy = abs(v_y - v_y_cmd)
      2: e_wz = abs(omega_z - omega_z_cmd)
      3: center-of-mass z-coordinate
      4: max(abs(roll), abs(pitch))
      5: number of currently contacting feet

    Active walk formula:
      G_[0,H](e_vx <= eps_vx and e_vy <= eps_vy and
               e_wz <= eps_wz and
               com_z >= z_min and
               max(abs(roll), abs(pitch)) <= theta_max and
               n_stance_contacts >= n_contact_min)

    Active trot formula:
      G_[0,H](e_vx <= eps_vx and e_vy <= eps_vy and
               e_wz <= eps_wz and
               com_z >= z_min and
               max(abs(roll), abs(pitch)) <= theta_max and
               n_stance_contacts >= n_contact_min)

    Active bound formula has the same six symbolic predicates, using thresholds
    selected universally across walk/trot/bound.  A real bound expert dataset
    was included in the gait-specific q50 analysis.

    Rolling yaw-rate and lateral-velocity specifications are intentionally
    active for every gait as documented overrides. Their raw q50 values and
    baseline masks remain recorded, while their dense-reward importance is
    reduced to one tenth of forward tracking. The baseline expert-data gate was
    applied independently for walk, trot, and bound after discarding timesteps
    0..49 from every trajectory.

    z_max and n_contact_max are intentionally disabled.

    rho_stl is the exact hard-min robustness used for satisfaction checks.
    rho_stl_smooth is normalized, importance-weighted smooth robustness used
    by the PPO reward. Forward tracking has weight 1.0; lateral and yaw
    tracking each have weight 0.1. Exact robustness remains unweighted.
    """
    if weights_override is not None:
        raise ValueError(
            "Subgroup reward-weight overrides are disabled; the reward uses "
            "the smooth robustness of the complete STL formula."
        )

    # Errors are precomputed in BarkourEnv, so commands are retained only for
    # API compatibility with the supplied project architecture.
    del commands

    stl = reward_input["stl_history"]
    e_vx_hist = stl[:, 0]
    e_vy_hist = stl[:, 1]
    e_wz_hist = stl[:, 2]
    com_z_hist = stl[:, 3]
    tilt_hist = stl[:, 4]
    n_contact_hist = stl[:, 5]

    # Exact predicate robustness in physical units.
    rho_vx_t = eps_vx - e_vx_hist
    rho_vy_t = eps_vy - e_vy_hist
    rho_wz_t = eps_wz - e_wz_hist
    rho_com_z_t = com_z_hist - z_min
    rho_tilt_t = theta_max - tilt_hist
    rho_contact_t = n_contact_hist - n_contact_min

    predicate_margins = jnp.stack(
        (
            rho_vx_t,
            rho_vy_t,
            rho_wz_t,
            rho_com_z_t,
            rho_tilt_t,
            rho_contact_t,
        ),
        axis=1,
    )

    walk_mask = jnp.asarray(WALK_ACTIVE_PREDICATE_MASK, dtype=bool)
    trot_mask = jnp.asarray(TROT_ACTIVE_PREDICATE_MASK, dtype=bool)
    bound_mask = jnp.asarray(BOUND_ACTIVE_PREDICATE_MASK, dtype=bool)
    active_predicate_mask = jnp.where(
        mode == MODE_WALK,
        walk_mask,
        jnp.where(mode == MODE_TROT, trot_mask, bound_mask),
    )
    active_predicate_mask_2d = active_predicate_mask[None, :]

    active_margins = jnp.where(
        active_predicate_mask_2d,
        predicate_margins,
        jnp.inf,
    )
    active_tracking_margins = jnp.where(
        active_predicate_mask_2d[:, :3],
        predicate_margins[:, :3],
        jnp.inf,
    )

    rho_tracking_t = jnp.min(active_tracking_margins, axis=1)
    rho_safety_t = jnp.minimum(rho_com_z_t, rho_tilt_t)
    rho_inst_t = jnp.min(active_margins, axis=1)

    # Exact quantitative robustness for diagnostics and satisfaction checks.
    rho_stl = _masked_min_last(rho_inst_t, valid_len)
    rho_tracking = _masked_min_last(rho_tracking_t, valid_len)
    rho_safety = _masked_min_last(rho_safety_t, valid_len)
    rho_contact = _masked_min_last(rho_contact_t, valid_len)
    rho_com_z = _masked_min_last(rho_com_z_t, valid_len)
    rho_tilt = _masked_min_last(rho_tilt_t, valid_len)

    # Config-derived normalization keeps heterogeneous physical units comparable.
    scales = jnp.asarray(
        (
            velocity_vx_margin_scale,
            velocity_vy_margin_scale,
            velocity_wz_margin_scale,
            com_z_margin_scale,
            tilt_margin_scale,
            min_contacts_margin_scale,
        ),
        dtype=jnp.float32,
    )
    normalized_margins = predicate_margins / jnp.maximum(scales, _EPS)
    predicate_reward_weights = jnp.asarray(
        PREDICATE_REWARD_WEIGHTS, dtype=jnp.float32
    )

    # Smooth conjunction followed by smooth G over the valid temporal suffix.
    smooth_inst_t = _smooth_min(
        normalized_margins,
        sharpness=beta,
        axis=1,
        mask=active_predicate_mask_2d,
        base_weights=predicate_reward_weights,
    )
    valid_mask = _valid_suffix_mask(smooth_inst_t.shape[0], valid_len)
    rho_stl_smooth = _smooth_min(
        smooth_inst_t,
        sharpness=beta,
        axis=0,
        mask=valid_mask,
    )

    tau = reward_input.get("tau_history")
    if tau is None:
        rho_torque = jnp.asarray(0.0, dtype=jnp.float32)
        tau_effort = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        tau_margin_worst_joint = jnp.min(tau_max - jnp.abs(tau), axis=1)
        rho_torque = _masked_min_last(tau_margin_worst_joint, valid_len)
        tau_sq_sum = jnp.sum(jnp.square(tau), axis=1)
        tau_effort = _masked_mean_last(tau_sq_sum, valid_len)

    reward = rho_stl_smooth - gamma_tau * tau_effort

    return (
        reward,
        rho_stl,
        rho_stl_smooth,
        rho_tracking,
        rho_safety,
        rho_contact,
        rho_com_z,
        rho_tilt,
        rho_torque,
        tau_effort,
    )
