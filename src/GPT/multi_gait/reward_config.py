from ml_collections import config_dict

from coeff_config import (
    c_walk_min,
    clearance_bound_min,
    clearance_trot_min,
    clearance_walk_min,
    duty_bound_max,
    duty_trot_max,
    duty_trot_min,
    duty_walk_min,
    e_vel_max,
    eps_diag_sync_trot_max,
    eps_front_hind_phase_bound_max,
    eps_front_sync_bound_max,
    eps_hind_sync_bound_max,
    h_base_min,
    p_2contact_trot_min,
    p_3plus_walk_min,
    p_diag2_trot_min,
    p_flight_bound_min,
    p_flight_walk_max,
    p_front_pair_bound_min,
    p_hind_pair_bound_min,
    slip_max,
    theta_pitch_max,
    theta_roll_max,
)


def get_config():
  """Returns original dense reward config for barkour quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    tracking_lin_vel=1.5,
                    tracking_ang_vel=0.8,
                    lin_vel_z=-2.0,
                    ang_vel_xy=-0.05,
                    orientation=-5.0,
                    torques=-0.0002,
                    action_rate=-0.01,
                    feet_air_time=0.2,
                    stand_still=-0.5,
                    termination=-1.0,
                    foot_slip=-0.1,
                )
            ),
            tracking_sigma=0.25,
        )
    )
    return default_config

  return config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )


def get_stl_config():
  """Returns STL reward config for barkour quadruped environment.

  Values are mapped only for the templates selected by GPT-5.5.  The two
  predicates without corresponding attached-config thresholds are explicitly
  disabled and retain None as a visible record of that decision.
  """

  def get_default_rewards_config():
    return config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    total_stl_reward=1.0,
                    rho_safety=1.0,
                    rho_tracking=1.0,
                    rho_gait=1.0,
                    rho_safety_raw=1.0,
                    rho_tracking_raw=1.0,
                    rho_gait_raw=1.0,
                    torque_lim=1.0,
                    torque_effort=1.0,
                    smooth_action=1.0,
                    early_termination_penalty=1.0,
                )
            ),
        )
    )

  # STL predicate parameters.
  #
  # Fill these from expert trajectory fitting.
  stl_params = config_dict.ConfigDict(
      dict(
          # Smooth STL robustness temperature.
          # This is a reward-shaping hyperparameter, not an STL threshold.
          softmin_beta=0.05,

          # Group reward weights and normalizers.
          # These can also be overridden through BarkourEnv(reward_weights=...).
          w_safe=1.0,
          alpha_safe=1.0,
          w_track=1.0,
          alpha_track=1.0,
          w_gait=1.0,
          alpha_gait=1.0,

          # Stage 2 jointly trains all three gait rewards.
          enabled_gait_modes=("walk", "trot", "bound"),

          # -------------------------------------------------------------------
          # Shared safety STL parameters.
          # Formula:
          # G_[0,H_safety](
          #   Roll <= theta_roll_max
          #   ∧ Pitch <= theta_pitch_max
          #   ∧ Base_height >= h_base_min
          #   ∧ Joint_torque <= tau_max
          #   ∧ Slip <= slip_max
          # )
          # tau_max is supplied in coeff_config.py.
          # -------------------------------------------------------------------
          theta_roll_max=theta_roll_max,
          theta_pitch_max=theta_pitch_max,
          h_base_min=h_base_min,
          # Disabled: the attached config has no upper base-height threshold.
          h_base_max=None,
          slip_max=slip_max,

          # -------------------------------------------------------------------
          # Shared tracking STL parameter.
          # Formula:
          # G_[0,H_track](Velocity_tracking_error <= e_vel_max)
          # -------------------------------------------------------------------
          e_vel_max=e_vel_max,

          # -------------------------------------------------------------------
          # Walk-like gait parameters.
          # Formula:
          # G_[0,H_walk](
          #   Contact_count >= c_walk_min
          #   ∧ P_3plus >= p_3plus_walk_min
          #   ∧ P_flight <= p_flight_walk_max
          #   ∧ Duty_factor >= duty_walk_min
          #   ∧ Swing_clearance >= clearance_walk_min
          # )
          # -------------------------------------------------------------------
          c_walk_min=c_walk_min,
          p_3plus_walk_min=p_3plus_walk_min,
          p_flight_walk_max=p_flight_walk_max,
          duty_walk_min=duty_walk_min,
          clearance_walk_min=clearance_walk_min,

          # -------------------------------------------------------------------
          # Trot-like gait parameters.
          # Formula:
          # G_[0,H_trot](
          #   P_diag2 >= p_diag2_trot_min
          #   ∧ P_2contact >= p_2contact_trot_min
          #   ∧ Diagonal_pair_sync_error <= eps_diag_sync_trot_max
          #   ∧ duty_trot_min <= Duty_factor <= duty_trot_max
          #   ∧ Swing_clearance >= clearance_trot_min
          # )
          # -------------------------------------------------------------------
          p_diag2_trot_min=p_diag2_trot_min,
          p_2contact_trot_min=p_2contact_trot_min,
          eps_diag_sync_trot_max=eps_diag_sync_trot_max,
          duty_trot_min=duty_trot_min,
          duty_trot_max=duty_trot_max,
          # Disabled: the attached config has no trot flight maximum.
          p_flight_trot_max=None,
          clearance_trot_min=clearance_trot_min,

          # -------------------------------------------------------------------
          # Bound-like gait parameters.
          # Formula:
          # G_[0,H_bound](
          #   Front_pair_sync_error <= eps_front_sync_bound_max
          #   ∧ Hind_pair_sync_error <= eps_hind_sync_bound_max
          #   ∧ Front_hind_phase_error <= eps_front_hind_phase_bound_max
          #   ∧ P_front_pair >= p_front_pair_bound_min
          #   ∧ P_hind_pair >= p_hind_pair_bound_min
          #   ∧ P_flight >= p_flight_bound_min
          #   ∧ Duty_factor <= duty_bound_max
          #   ∧ Swing_clearance >= clearance_bound_min
          # )
          # -------------------------------------------------------------------
          eps_front_sync_bound_max=eps_front_sync_bound_max,
          eps_hind_sync_bound_max=eps_hind_sync_bound_max,
          eps_front_hind_phase_bound_max=eps_front_hind_phase_bound_max,
          p_front_pair_bound_min=p_front_pair_bound_min,
          p_hind_pair_bound_min=p_hind_pair_bound_min,
          p_flight_bound_min=p_flight_bound_min,
          duty_bound_max=duty_bound_max,
          clearance_bound_min=clearance_bound_min,
      )
  )

  return config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
          stl=stl_params,
      )
  )
