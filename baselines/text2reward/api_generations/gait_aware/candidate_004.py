from __future__ import annotations
import jax
import jax.numpy as jnp
from reward_context import Text2RewardContext


def compute_dense_reward(
    ctx: Text2RewardContext,
) -> jax.Array:
    # -------------------------------------------------------------------------
    # Command tracking.
    # Use one unified normalized tracking error so that, for example, matching
    # zero lateral/yaw commands does not dominate the reward when forward speed
    # tracking is poor.
    # -------------------------------------------------------------------------
    fwd_err_sq = jnp.square(ctx.base_lin_vel[0] - ctx.command[0])
    lat_err_sq = jnp.square(ctx.base_lin_vel[1] - ctx.command[1])
    yaw_err_sq = jnp.square(ctx.base_ang_vel[2] - ctx.command[2])

    fwd_scale_sq = 0.35 * 0.35
    lat_scale_sq = 0.30 * 0.30
    yaw_scale_sq = 0.50 * 0.50

    normalized_tracking_error = (
        fwd_err_sq / fwd_scale_sq
        + lat_err_sq / lat_scale_sq
        + yaw_err_sq / yaw_scale_sq
    )

    unified_tracking_reward = 5.0 / (1.0 + normalized_tracking_error)

    # Small axis-wise bonuses keep each command dimension individually shaped.
    fwd_tracking_reward = 0.8 / (1.0 + fwd_err_sq / fwd_scale_sq)
    lat_tracking_reward = 0.5 / (1.0 + lat_err_sq / lat_scale_sq)
    yaw_tracking_reward = 0.7 / (1.0 + yaw_err_sq / yaw_scale_sq)

    tracking_reward = (
        unified_tracking_reward
        + fwd_tracking_reward
        + lat_tracking_reward
        + yaw_tracking_reward
    )

    # -------------------------------------------------------------------------
    # Stability rewards.
    # projected_gravity should be close to [0, 0, -1] when upright.
    # -------------------------------------------------------------------------
    upright_error = (
        jnp.square(ctx.projected_gravity[0])
        + jnp.square(ctx.projected_gravity[1])
        + 0.25 * jnp.square(ctx.projected_gravity[2] + 1.0)
    )
    upright_reward = jnp.exp(-upright_error / 0.25)

    # Broad Barkour nominal torso height target.
    height_reward = jnp.exp(-jnp.square(ctx.base_height - 0.34) / 0.0064)

    vertical_velocity_reward = jnp.exp(-jnp.square(ctx.base_lin_vel[2]) / 0.25)

    roll_pitch_rate_reward = jnp.exp(
        -jnp.sum(jnp.square(ctx.base_ang_vel[:2])) / 2.25
    )

    stability_reward = (
        1.0 * upright_reward
        + 0.25 * height_reward
        + 0.25 * vertical_velocity_reward
        + 0.25 * roll_pitch_rate_reward
    )

    # -------------------------------------------------------------------------
    # Smooth gait selection from commanded forward speed.
    # Low: walking-trot/support-rich.
    # Mid: diagonal trot.
    # High: bound.
    # -------------------------------------------------------------------------
    cmd_forward_speed = jnp.maximum(ctx.command[0], 0.0)
    transition_width = 0.12

    low_gate = 1.0 / (
        1.0 + jnp.exp((cmd_forward_speed - 0.7) / transition_width)
    )
    mid_gate = (
        1.0 / (1.0 + jnp.exp(-(cmd_forward_speed - 0.7) / transition_width))
    ) * (
        1.0 / (1.0 + jnp.exp((cmd_forward_speed - 1.7) / transition_width))
    )
    high_gate = 1.0 / (
        1.0 + jnp.exp(-(cmd_forward_speed - 1.7) / transition_width)
    )

    gate_sum = low_gate + mid_gate + high_gate + 1.0e-6
    walk_weight = low_gate / gate_sum
    trot_weight = mid_gate / gate_sum
    bound_weight = high_gate / gate_sum

    # Contacts: [front_left, hind_left, front_right, hind_right].
    contact = jnp.asarray(ctx.foot_contact, dtype=jnp.float32)
    fl = contact[0]
    hl = contact[1]
    fr = contact[2]
    hr = contact[3]
    contact_count = jnp.sum(contact)

    # -------------------------------------------------------------------------
    # Low-speed support-rich walking-trot gait.
    # Near zero command, four-foot support is acceptable. As speed increases
    # toward 0.7 m/s, prefer roughly three supporting feet.
    # -------------------------------------------------------------------------
    near_zero_speed = jnp.exp(-jnp.square(cmd_forward_speed) / 0.04)
    walk_target_contacts = 3.0 + near_zero_speed
    walk_contact_reward = jnp.exp(
        -jnp.square(contact_count - walk_target_contacts) / 2.0
    )

    # Diagonal compatibility is gentle here, because the desired gait is
    # support-rich rather than a strict two-beat trot.
    walk_diagonal_compatibility = jnp.exp(
        -(jnp.square(fl - hr) + jnp.square(fr - hl)) / 2.0
    )

    walk_gait_reward = (
        0.65 * walk_contact_reward
        + 0.35 * walk_diagonal_compatibility
    )

    # -------------------------------------------------------------------------
    # Medium-speed diagonal trot.
    # Desired stance patterns are FL+HR or FR+HL.
    # -------------------------------------------------------------------------
    diagonal_pair_sync = jnp.exp(
        -(jnp.square(fl - hr) + jnp.square(fr - hl)) / 0.5
    )

    diag_a = 0.5 * (fl + hr)
    diag_b = 0.5 * (fr + hl)

    diagonal_alternation = jnp.square(diag_a - diag_b)
    trot_contact_count_reward = jnp.exp(-jnp.square(contact_count - 2.0) / 1.0)

    # Small allowance for aerial phases in a dynamic trot.
    flight_reward = jnp.exp(-jnp.square(contact_count) / 0.25)

    trot_gait_reward = diagonal_pair_sync * (
        0.75 * diagonal_alternation
        + 0.25 * trot_contact_count_reward
        + 0.05 * flight_reward
    )

    # -------------------------------------------------------------------------
    # High-speed bound.
    # Desired stance patterns are front-pair support or hind-pair support.
    # -------------------------------------------------------------------------
    front_hind_pair_sync = jnp.exp(
        -(jnp.square(fl - fr) + jnp.square(hl - hr)) / 0.5
    )

    front_mean = 0.5 * (fl + fr)
    hind_mean = 0.5 * (hl + hr)

    front_hind_alternation = jnp.square(front_mean - hind_mean)
    bound_contact_count_reward = jnp.exp(-jnp.square(contact_count - 2.0) / 1.0)

    # Bounds commonly include a brief aerial phase, so give it a small reward.
    bound_gait_reward = front_hind_pair_sync * (
        0.80 * front_hind_alternation
        + 0.20 * bound_contact_count_reward
        + 0.10 * flight_reward
    )

    gait_reward = (
        walk_weight * walk_gait_reward
        + trot_weight * trot_gait_reward
        + bound_weight * bound_gait_reward
    )

    # -------------------------------------------------------------------------
    # Foot quality rewards.
    # Stance feet should have low world-frame horizontal velocity.
    # Swing feet should clear the ground slightly.
    # -------------------------------------------------------------------------
    foot_horizontal_speed_sq = jnp.sum(
        jnp.square(ctx.foot_velocity_world[:, :2]), axis=1
    )
    mean_stance_slip_sq = (
        jnp.sum(contact * foot_horizontal_speed_sq) / (contact_count + 1.0e-6)
    )
    foot_slip_reward = jnp.exp(-mean_stance_slip_sq / 0.25)

    swing = 1.0 - contact
    swing_count = jnp.sum(swing)
    clearance_deficit = jnp.maximum(0.0, 0.06 - ctx.foot_position_world[:, 2])
    mean_clearance_deficit_sq = (
        jnp.sum(swing * jnp.square(clearance_deficit)) / (swing_count + 1.0e-6)
    )
    swing_clearance_reward = jnp.exp(-mean_clearance_deficit_sq / 0.0025)

    foot_quality_reward = (
        0.35 * foot_slip_reward
        + 0.15 * swing_clearance_reward
    )

    # -------------------------------------------------------------------------
    # Small regularization terms.
    # These should not dominate the task reward.
    # -------------------------------------------------------------------------
    joint_pos_penalty = 0.02 * jnp.mean(jnp.square(ctx.joint_pos_error))
    joint_vel_penalty = 0.0005 * jnp.mean(jnp.square(ctx.joint_vel))
    torque_penalty = 0.00002 * jnp.mean(jnp.square(ctx.motor_torque))
    action_rate_penalty = 0.04 * jnp.mean(
        jnp.square(ctx.action - ctx.previous_action)
    )
    action_magnitude_penalty = 0.01 * jnp.mean(jnp.square(ctx.action))

    regularization_penalty = (
        joint_pos_penalty
        + joint_vel_penalty
        + torque_penalty
        + action_rate_penalty
        + action_magnitude_penalty
    )

    done_penalty = 5.0 * jnp.asarray(ctx.done, dtype=jnp.float32)

    reward = (
        tracking_reward
        + stability_reward
        + 1.8 * gait_reward
        + foot_quality_reward
        - regularization_penalty
        - done_penalty
    )

    return reward
