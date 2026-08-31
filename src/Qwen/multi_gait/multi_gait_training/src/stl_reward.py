from pathlib import Path
import sys

# Also allow this reward module to be imported independently of BarkourEnv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'configs'))

import jax
import jax.numpy as jnp

from coeff_config import (
    tau_max,
    beta,
    beta_safe,
    beta_timing,
    beta_pattern,
    gamma_tau,
    eps_vx_by_mode,
    eps_vy_by_mode,
    eps_yaw_by_mode,
    min_contacts_by_mode,
    com_z_by_mode,
    abs_vz_by_mode,
    roll_abs_by_mode,
    pitch_abs_by_mode,
    slip_speed_by_mode,
    cop_com_xy_dist_by_mode,
    stride_period_by_mode,
    duty_factor_by_mode,
    diag_phase_error_by_mode,
    diag_2contact_fraction_min_by_mode,
    diag_2contact_fraction_max_by_mode,
    contact2_fraction_min_by_mode,
    flight_fraction_min_by_mode,
    front_only_fraction_min_by_mode,
    hind_only_fraction_min_by_mode,
    front_only_fraction_max_by_mode,
    hind_only_fraction_max_by_mode,
    all4_fraction_max_by_mode,
    all4_fraction_min_by_mode,
    pair_front_mismatch_max_by_mode,
    pair_hind_mismatch_max_by_mode,
    hind_to_front_lag_by_mode,
    K_REQUIRE_3PLUS,
    H_by_mode,
    MODE_BOUND,
    MODE_TROT,
    MODE_WALK,
    H_WARMUP_MIN_VALID,
    DT,
    clearance_min_by_mode,
    tau_margin_scale,
    min_contacts_margin_scale,
    com_z_margin_scale,
    abs_vz_margin_scale,
    roll_margin_scale_deg,
    pitch_margin_scale_deg,
    slip_margin_scale,
    support_margin_scale,
    track_axis_weights,
    stride_margin_scale_by_mode,
    duty_margin_scale_by_mode,
    diag_phase_margin_scale_by_mode,
    diag2_margin_scale_by_mode,
    contact2_margin_scale_by_mode,
    pair_mismatch_margin_scale_by_mode,
    hindfront_margin_scale_by_mode,
    flight_margin_scale_by_mode,
    front_only_margin_scale_by_mode,
    hind_only_margin_scale_by_mode,
    all4_margin_scale_by_mode,
    event3plus_margin_scale,
    bound_event_margin_scale,
    clearance_margin_scale_by_mode,
    w_safe_by_mode,
    w_track_by_mode,
    w_timing_by_mode,
    w_pattern_by_mode,
    alpha_safe_by_mode,
    alpha_track_by_mode,
    alpha_timing_by_mode,
    alpha_pattern_by_mode,
)


def tanh_norm(rho, alpha):
    return jnp.tanh(rho / jnp.maximum(alpha, 1e-6))


def smooth_min(vals, beta=10.0):
    vals = jnp.asarray(vals, dtype=jnp.float32)
    return -jax.nn.logsumexp(-beta * vals) / beta


def smooth_min_sign_preserving(vals, beta=10.0):
    vals = jnp.asarray(vals, dtype=jnp.float32)
    w = jax.nn.softmax(-beta * vals)
    return jnp.sum(w * vals)


def _window_mask(H, valid_len, horizon):
    """Mask the last `active_len=min(valid_len,horizon)` entries of right-aligned history."""
    active_len = jnp.minimum(jnp.minimum(valid_len, horizon), H)
    idx = jnp.arange(H)
    return idx >= (H - active_len)


def _masked_min(x, mask):
    x = jnp.asarray(x)
    m = jnp.asarray(mask)
    while m.ndim < x.ndim:
        m = m[..., None]
    x_masked = jnp.where(m, x, jnp.inf)
    return jnp.min(x_masked)


def _masked_mean(x, mask):
    x = jnp.asarray(x)
    m = jnp.asarray(mask, dtype=x.dtype)
    while m.ndim < x.ndim:
        m = m[..., None]
    denom = jnp.maximum(jnp.sum(m), 1.0)
    return jnp.sum(x * m) / denom


def _safe_div(x, s):
    return x / jnp.maximum(s, 1e-6)


def _interval_robustness(x, lo, hi):
    return jnp.minimum(x - lo, hi - x)


def _touchdown_events(signal, mask):
    signal = signal > 0.5
    prev = jnp.concatenate([jnp.array([False]), signal[:-1]])
    idx = jnp.arange(signal.shape[0])
    first_valid = jnp.argmax(mask.astype(jnp.int32))
    prev = jnp.where(idx == first_valid, False, prev)
    return signal & (~prev) & mask


def _event_count(events):
    return jnp.sum(events.astype(jnp.float32))


def _mean_period_seconds(event_counts, valid_steps):
    valid_steps = jnp.maximum(valid_steps, 1.0)
    has_evt = event_counts > 0.0
    period_steps = valid_steps / jnp.maximum(event_counts, 1.0)
    denom = jnp.maximum(jnp.sum(has_evt.astype(jnp.float32)), 1.0)
    return jnp.sum(jnp.where(has_evt, period_steps, 0.0)) / denom * DT


def _inphase_event_error(events_a, events_b, period_steps):
    period_steps = jnp.maximum(period_steps, 1.0)
    H = events_a.shape[0]
    idx = jnp.arange(H)
    ia = idx[:, None]
    ib = idx[None, :]
    dist = jnp.abs(ia - ib).astype(jnp.float32)
    valid = events_a[:, None] & events_b[None, :]
    min_d = jnp.min(jnp.where(valid, dist, jnp.inf), axis=1)
    frac = min_d / period_steps
    frac = jnp.minimum(frac, jnp.abs(1.0 - frac))
    select = events_a & jnp.isfinite(min_d)
    denom = jnp.maximum(jnp.sum(select.astype(jnp.float32)), 1.0)
    err = jnp.sum(jnp.where(select, frac, 0.0)) / denom
    return jnp.where(jnp.any(select), err, 1.0)


def _forward_lag(events_src, events_dst, period_steps):
    period_steps = jnp.maximum(period_steps, 1.0)
    H = events_src.shape[0]
    idx = jnp.arange(H)
    src = idx[:, None]
    dst = idx[None, :]
    d = (dst - src).astype(jnp.float32)
    valid = events_src[:, None] & events_dst[None, :] & (d >= 0.0)
    min_fwd = jnp.min(jnp.where(valid, d, jnp.inf), axis=1)
    select = events_src & jnp.isfinite(min_fwd)
    denom = jnp.maximum(jnp.sum(select.astype(jnp.float32)), 1.0)
    lag = jnp.sum(jnp.where(select, min_fwd / period_steps, 0.0)) / denom
    return jnp.where(jnp.any(select), lag, -1.0)


def _mean_pair_mismatch(sig_a, sig_b, mask):
    mismatch = jnp.abs(sig_a.astype(jnp.float32) - sig_b.astype(jnp.float32))
    return _masked_mean(mismatch, mask)


def _optional_clearance_robustness(reward_input, contacts, mask, mode):
    clearance_hist = reward_input.get("clearance_history", None)
    if clearance_hist is None:
        return jnp.array(0.0, dtype=jnp.float32)

    clearance_hist = jnp.asarray(clearance_hist, dtype=jnp.float32)
    swing_mask = mask[:, None] & (contacts < 0.5)
    cmin = jnp.asarray(clearance_min_by_mode)[mode]
    margins = clearance_hist - cmin
    any_swing = jnp.any(swing_mask)
    return jnp.where(any_swing, _masked_min(margins, swing_mask), 0.0)


def _default_group_params(mode):
    mode = jnp.asarray(mode, dtype=jnp.int32)
    return (
        jnp.asarray(w_track_by_mode)[mode],
        jnp.asarray(w_safe_by_mode)[mode],
        jnp.asarray(w_timing_by_mode)[mode],
        jnp.asarray(w_pattern_by_mode)[mode],
        jnp.asarray(alpha_track_by_mode)[mode],
        jnp.asarray(alpha_safe_by_mode)[mode],
        jnp.asarray(alpha_timing_by_mode)[mode],
        jnp.asarray(alpha_pattern_by_mode)[mode],
        jnp.asarray(beta, dtype=jnp.float32),
    )


def reward_step(reward_input, commands, mode, valid_len, weights_override=None):
    tau = reward_input["tau_history"]                          # (H, n_tau)
    c = reward_input["contact_history"].astype(jnp.float32)    # (H,4)
    feet_xy = reward_input["feet_history"]                     # (H,4,2)
    com_xy = reward_input["CoM_history"]                       # (H,2)
    v_hist = reward_input["lin_velocity_history"]              # (H,3)
    yaw_hist = reward_input["ang_velocity_history"]            # (H,)
    com_z_hist = reward_input["com_z_history"]                 # (H,)
    roll_hist = reward_input["roll_history"]                   # (H,)
    pitch_hist = reward_input["pitch_history"]                 # (H,)
    slip_hist = reward_input["slipmax_history"]                # (H,)

    if weights_override is None:
        (
            _w_track,
            _w_safe,
            _w_timing,
            _w_pattern,
            _alpha_track,
            _alpha_safe,
            _alpha_timing,
            _alpha_pattern,
            _beta_override,
        ) = _default_group_params(mode)
    else:
        (
            _w_track,
            _w_safe,
            _w_timing,
            _w_pattern,
            _alpha_track,
            _alpha_safe,
            _alpha_timing,
            _alpha_pattern,
            _beta_override,
        ) = weights_override

    H = tau.shape[0]
    valid_len = jnp.minimum(valid_len, H)
    horizon = jnp.asarray(H_by_mode)[mode]
    mask = _window_mask(H, valid_len, horizon)
    active_steps = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    gait_enabled = valid_len >= H_WARMUP_MIN_VALID

    v_x_star = commands[0]
    v_y_star = commands[1]
    yaw_star = commands[2]

    FL, HL, FR, HR = 0, 1, 2, 3
    current_contacts = c[-1]

    # ------------------------------------------------------------------
    # Shared safety terms - implementing STL predicates from JSON
    # ------------------------------------------------------------------
    # roll_bound: Roll <= roll_max
    roll_max_deg = jnp.asarray(roll_abs_by_mode)[mode]
    roll_max_rad = roll_max_deg * jnp.pi / 180.0
    rho_roll = _masked_min(roll_max_rad - jnp.abs(roll_hist), mask)

    # pitch_bound: Pitch <= pitch_max
    pitch_max_deg = jnp.asarray(pitch_abs_by_mode)[mode]
    pitch_max_rad = pitch_max_deg * jnp.pi / 180.0
    rho_pitch = _masked_min(pitch_max_rad - jnp.abs(pitch_hist), mask)

    # One-sided height predicate: base_height >= h_min.
    base_height_min_universal = 0.16
    rho_comz = _masked_min(com_z_hist - base_height_min_universal, mask)

    # slip_bound: Slip <= slip_max
    slip_max = jnp.asarray(slip_speed_by_mode)[mode]
    rho_slip = _masked_min(slip_max - slip_hist, mask)

    # swing_clearance_bound: Swing_clearance >= swing_clearance_min
    # Note: clearance_history not available in current reward_input, so this term is disabled
    rho_clearance = jnp.array(0.0, dtype=jnp.float32)

    # torque limits (additional safety)
    tau_margin = tau_max - jnp.abs(tau)
    tau_margin_worst_joint = jnp.min(tau_margin, axis=1)
    rho_torque = _masked_mean(tau_margin_worst_joint, mask)

    # q50 component filter: slip is retained for trot (q50 > 0), but remains
    # excluded for walk and bound.  Height remains excluded in every mode.
    safety_base = [
        _safe_div(rho_torque, tau_margin_scale),
        _safe_div(rho_roll, roll_margin_scale_deg * jnp.pi / 180.0),
        _safe_div(rho_pitch, pitch_margin_scale_deg * jnp.pi / 180.0),
    ]
    rho_safety_without_slip = smooth_min_sign_preserving(
        safety_base, beta=beta_safe
    )
    rho_safety_with_slip = smooth_min_sign_preserving(
        safety_base + [_safe_div(rho_slip, slip_margin_scale)], beta=beta_safe
    )
    rho_safety = jnp.where(
        mode == MODE_TROT, rho_safety_with_slip, rho_safety_without_slip
    )

    # ------------------------------------------------------------------
    # Tracking terms - implementing STL predicate from JSON
    # ------------------------------------------------------------------
    # velocity_tracking_error_bound: Velocity_tracking_error <= vel_err_max
    # Using vx error as primary tracking error
    eps_vx = jnp.asarray(eps_vx_by_mode)[mode]
    v_x_error_hist = jnp.abs(v_hist[:, 0] - v_x_star)
    rho_v_x_error = _masked_min(eps_vx - v_x_error_hist, mask)

    # For universal vel_err_max, use max across modes: max(0.55, 0.60, 0.60) = 0.60
    vel_err_max_universal = 0.60
    rho_vel_tracking_universal = _masked_min(vel_err_max_universal - v_x_error_hist, mask)

    rho_tracking = rho_vel_tracking_universal

    # ------------------------------------------------------------------
    # Windowed gait features for mode-specific specifications
    # ------------------------------------------------------------------
    c01 = c.astype(jnp.int32)

    td_FL = _touchdown_events(c01[:, FL], mask)
    td_HL = _touchdown_events(c01[:, HL], mask)
    td_FR = _touchdown_events(c01[:, FR], mask)
    td_HR = _touchdown_events(c01[:, HR], mask)

    n_contacts = jnp.sum(c, axis=1)

    front_pair = (c01[:, FL] == 1) & (c01[:, FR] == 1)
    hind_pair = (c01[:, HL] == 1) & (c01[:, HR] == 1)
    front_only = front_pair & (~hind_pair)
    hind_only = hind_pair & (~front_pair)
    all4 = front_pair & hind_pair
    flight = n_contacts == 0.0
    diag2_pattern = (((c01[:, FL] == 1) & (c01[:, HR] == 1) & (c01[:, HL] == 0) & (c01[:, FR] == 0))
                    | ((c01[:, FR] == 1) & (c01[:, HL] == 1) & (c01[:, FL] == 0) & (c01[:, HR] == 0)))

    p_3plus = _masked_mean((n_contacts >= 3.0).astype(jnp.float32), mask)
    p_flight = _masked_mean(flight.astype(jnp.float32), mask)
    p_diag2 = _masked_mean(diag2_pattern.astype(jnp.float32), mask)
    p_2contact = _masked_mean((n_contacts == 2.0).astype(jnp.float32), mask)
    p_front_only = _masked_mean(front_only.astype(jnp.float32), mask)
    p_hind_only = _masked_mean(hind_only.astype(jnp.float32), mask)

    # Duty factor calculation
    duty_leg = jnp.sum(c * mask[:, None].astype(jnp.float32), axis=0) / active_steps
    duty_est = jnp.mean(duty_leg)

    # Diagonal sync error
    td_counts = jnp.array([
        _event_count(td_FL),
        _event_count(td_HL),
        _event_count(td_FR),
        _event_count(td_HR),])
    stride_sec_est = _mean_period_seconds(td_counts, active_steps)
    period_steps = jnp.maximum(stride_sec_est / DT, 1.0)
    e_diag1 = _inphase_event_error(td_FL, td_HR, period_steps)
    e_diag2 = _inphase_event_error(td_FR, td_HL, period_steps)
    e_diag = 0.5 * (e_diag1 + e_diag2)
    td_front_pair = _touchdown_events(front_pair, mask)
    td_hind_pair = _touchdown_events(hind_pair, mask)
    hind_to_front_lag = _forward_lag(td_hind_pair, td_front_pair, period_steps)

    # ------------------------------------------------------------------
    # Mode-specific STL specifications with numerical parameters
    # ------------------------------------------------------------------
    # Walk-like specifications
    # The generated signal is a fraction, hence 11/30.
    p_3plus_min_walk = K_REQUIRE_3PLUS / H_by_mode[MODE_WALK]
    p_flight_max_walk = 0.01  # Near zero flight time
    duty_factor_min_walk = 0.62  # From duty_factor_by_mode[0][0]

    rho_three_plus_contact = p_3plus - p_3plus_min_walk
    rho_no_flight = p_flight_max_walk - p_flight
    rho_duty_factor_high = duty_est - duty_factor_min_walk

    rho_walk_stl = smooth_min_sign_preserving(
        [
            rho_no_flight,
            rho_duty_factor_high
        ],
        beta=beta_pattern,
    )

    # Trot-like specifications
    p_diag2_min_trot = 0.99  # From diag_2contact_fraction_min_by_mode[1]
    p_2contact_min_trot = 0.70  # From contact2_fraction_min_by_mode[1]
    diag_sync_err_max_trot = 0.11  # From diag_phase_error_by_mode[1]
    p_flight_trot_max = 0.10  # Heuristic: allow brief flight phases

    rho_diagonal_pair_support = p_diag2 - p_diag2_min_trot
    rho_two_contact_dominance = p_2contact - p_2contact_min_trot
    rho_diagonal_sync_error = diag_sync_err_max_trot - e_diag
    rho_low_flight_time = p_flight_trot_max - p_flight

    rho_trot_stl = smooth_min_sign_preserving(
        [
            rho_two_contact_dominance,
            rho_diagonal_sync_error,
            rho_low_flight_time
        ],
        beta=beta_pattern,
    )

    # Bound-like specifications mapped directly from the reference config.
    pair_support_min = (
        front_only_fraction_min_by_mode[MODE_BOUND]
        + hind_only_fraction_min_by_mode[MODE_BOUND]
    )
    rho_pair_support_bound = p_front_only + p_hind_only - pair_support_min
    rho_flight_bound = p_flight - flight_fraction_min_by_mode[MODE_BOUND]
    phase_lo, phase_hi = hind_to_front_lag_by_mode[MODE_BOUND]
    rho_phase_bound = jnp.where(
        hind_to_front_lag >= 0.0,
        _interval_robustness(hind_to_front_lag, phase_lo, phase_hi),
        0.0,
    )
    rho_low_diag_bound = (
        diag_2contact_fraction_max_by_mode[MODE_BOUND] - p_diag2
    )
    rho_bound_stl = smooth_min_sign_preserving(
        [
            rho_pair_support_bound,
            rho_flight_bound,
            rho_phase_bound,
            rho_low_diag_bound,
        ],
        beta=beta_pattern,
    )

    # Select mode-specific gait robustness
    rho_gait_pattern = jnp.where(
        mode == MODE_WALK,
        rho_walk_stl,
        jnp.where(mode == MODE_TROT, rho_trot_stl, rho_bound_stl),
    )
    rho_gait_pattern = jnp.where(gait_enabled, rho_gait_pattern, 0.0)

    # Timing robustness (stride and duty factor within bounds)
    stride_bounds = jnp.asarray(stride_period_by_mode)
    duty_bounds = jnp.asarray(duty_factor_by_mode)

    stride_lo = stride_bounds[mode, 0]
    stride_hi = stride_bounds[mode, 1]
    duty_lo = duty_bounds[mode, 0]
    duty_hi = duty_bounds[mode, 1]

    rho_stride = _interval_robustness(stride_sec_est, stride_lo, stride_hi)
    rho_duty = _interval_robustness(duty_est, duty_lo, duty_hi)

    stride_scale = jnp.asarray(stride_margin_scale_by_mode)[mode]
    duty_scale = jnp.asarray(duty_margin_scale_by_mode)[mode]

    rho_timing = smooth_min_sign_preserving(
        [
            _safe_div(rho_stride, stride_scale),
            _safe_div(rho_duty, duty_scale),
        ],
        beta=beta_timing,
    )
    rho_timing = jnp.where(gait_enabled, rho_timing, 0.0)

    # Combined gait robustness
    rho_gait = smooth_min_sign_preserving([rho_timing, rho_gait_pattern], beta=_beta_override)
    rho_gait = jnp.where(gait_enabled, rho_gait, 0.0)

    # Torque effort regularizer
    tau_sq_sum = jnp.sum(jnp.square(tau), axis=1)
    tau_effort = _masked_mean(tau_sq_sum, mask)

    # Final reward combining all STL components
    r = ( _w_safe * tanh_norm(rho_safety, _alpha_safe)
        + _w_track * tanh_norm(rho_tracking, _alpha_track)
        + _w_pattern * tanh_norm(rho_gait_pattern, _alpha_pattern)
        - gamma_tau * tau_effort
        )

    return (
        r,
        tau_effort,
        rho_safety,
        rho_torque,
        rho_comz,
        rho_roll,
        rho_pitch,
        rho_slip,
        rho_bound_stl,
        rho_trot_stl,
        rho_walk_stl,
        rho_v_x_error,
        _masked_min(jnp.abs(v_hist[:, 1] - v_y_star), mask),  # rho_v_y_error placeholder
        _masked_min(jnp.abs(yaw_hist - yaw_star), mask),      # rho_yaw_error placeholder
        _masked_min(n_contacts - 2.0, mask),                  # rho_nlegs placeholder
        rho_gait,
        _masked_mean(v_x_error_hist, mask),
        _masked_mean(jnp.abs(v_hist[:, 1] - v_y_star), mask),
        _masked_mean(jnp.abs(yaw_hist - yaw_star), mask),
        jnp.where(mode == MODE_BOUND, rho_low_diag_bound, rho_diagonal_pair_support),
        rho_stride,
        rho_duty,
        rho_three_plus_contact,
        jnp.array(0.0),  # rho_support placeholder
        rho_diagonal_sync_error,
        rho_two_contact_dominance,
        rho_tracking,
        rho_timing,
        rho_gait_pattern,
        p_front_only - front_only_fraction_min_by_mode[MODE_BOUND],
        p_hind_only - hind_only_fraction_min_by_mode[MODE_BOUND],
        rho_phase_bound,
        jnp.where(mode == MODE_BOUND, rho_flight_bound, rho_low_flight_time),
        p_front_only - front_only_fraction_min_by_mode[MODE_BOUND],
        p_hind_only - hind_only_fraction_min_by_mode[MODE_BOUND],
        jnp.array(0.0),  # rho_all4 placeholder
        jnp.array(0.0),  # rho_bound_event placeholder
        pitch_hist[-1],
        roll_hist[-1],
        current_contacts[0],
        current_contacts[1],
        current_contacts[2],
        current_contacts[3],
    )
