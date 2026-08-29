# =========================
# Barkour STL reward config
# =========================

# Barkour vb joint torque limit and joint velocity limit
# reward_step only uses tau_max directly. q_dot_max is kept for compatibility.
tau_max = 18.0  # Nm
q_dot_max = 25.0  # rad/s

# -----------------------------------------------------------------------------
# Temporal settings
# -----------------------------------------------------------------------------
DT = 0.02  # 50 Hz control step
H = 20
H_MAX = H
H_by_mode = (30, 24, 24)  # walk, trot, bound
H_WARMUP_MIN_VALID = 8

# -----------------------------------------------------------------------------
# Gait mode IDs
# -----------------------------------------------------------------------------
MODE_WALK = 0
MODE_TROT = 1
MODE_BOUND = 2

# Stage 2 curriculum: warm-start the selected walk+trot checkpoint and
# jointly train walk, trot, and bound for 400M additional environment steps.
ENABLE_BOUND_GAIT = True
TRAINING_GAIT_MODES = (MODE_WALK, MODE_TROT, MODE_BOUND)
TRAINING_COMMAND_SAMPLING = "mixed"
TRAINING_BOUND_STRAIGHT_COMMANDS = True
TRAINING_OBS_NOISE = 0.001
TRAINING_KICK_VEL = 0.001
TRAINING_NUM_TIMESTEPS = 400_000_000
VISUAL_EVAL_SPEEDS = (0.3, 0.5, 0.9, 1.0, 1.2)

# PPO survival safeguard.  This is not an STL predicate: it is an unscaled
# cost applied only when the robot physically fails before the episode timeout.
EARLY_TERMINATION_PENALTY = 1.6

# -----------------------------------------------------------------------------
# Command-speed hysteresis on |vx| [m/s]
# -----------------------------------------------------------------------------
WALK_TO_TROT_ENTER = 0.72
TROT_TO_WALK_EXIT = 0.65
TROT_TO_BOUND_ENTER = 1.55
BOUND_TO_TROT_EXIT = 1.45

# Mixed-regime sampling and bound-focused curriculum helpers.
regime_sample_probs = (0.25, 0.35, 0.40)
cmd_vx_range = (0.0, 1.9)
cmd_vy_range = (-0.2, 0.2)
cmd_yaw_range = (-0.2, 0.2)
bound_vx_sample_range = (1.55, 1.9)

# Robust aggregation sharpness.
beta = 0.5
beta_safe = 0.5
beta_timing = 0.5
beta_pattern = 0.5

# -----------------------------------------------------------------------------
# Mode-conditioned tracking tolerances
# -----------------------------------------------------------------------------
eps_vx_by_mode = (0.55, 0.60, 0.60)
eps_vy_by_mode = (0.05, 0.05, 0.05)
eps_yaw_by_mode = (0.05, 0.05, 0.05)

# -----------------------------------------------------------------------------
# Shared safety thresholds (walk, trot, bound)
# -----------------------------------------------------------------------------
min_contacts_by_mode = (2, 2, 0)
min_contacts = 2
abs_vz_by_mode = (0.15, 0.18, 0.25)
com_z_by_mode = (0.18, 0.22, 0.16)
cop_com_xy_dist_by_mode = (0.10, 0.13, 0.15)
roll_abs_by_mode = (10.0, 7.0, 15.0)
pitch_abs_by_mode = (8.0, 7.0, 15.0)
slip_speed_by_mode = (0.40, 0.75, 0.85)

# -----------------------------------------------------------------------------
# Mode-dependent gait structure parameters
# -----------------------------------------------------------------------------
stride_period_by_mode = ((0.42, 0.54), (0.33, 0.43), (0.25, 0.31))
duty_factor_by_mode = ((0.62, 0.74), (0.52, 0.60), (0.62, 0.70))

diag_phase_error_by_mode = (0.10, 0.11, 1.00)
diag_2contact_fraction_min_by_mode = (0.92, 0.99, 0.00)
diag_2contact_fraction_max_by_mode = (1.00, 1.00, 0.03)
contact2_fraction_min_by_mode = (0.00, 0.70, 0.00)

flight_fraction_min_by_mode = (0.00, 0.00, 0.04)
front_only_fraction_min_by_mode = (0.00, 0.00, 0.18)
hind_only_fraction_min_by_mode = (0.00, 0.00, 0.16)
front_only_fraction_max_by_mode = (1.00, 1.00, 0.42)
hind_only_fraction_max_by_mode = (1.00, 1.00, 0.38)

all4_fraction_max_by_mode = (1.00, 0.10, 0.30)
all4_fraction_min_by_mode = (0.00, 0.50, 0.10)

pair_front_mismatch_max_by_mode = (1.00, 1.00, 0.10)
pair_hind_mismatch_max_by_mode = (1.00, 1.00, 0.14)
hind_to_front_lag_by_mode = ((0.00, 1.00), (0.00, 1.00), (0.33, 0.40))

# Compatibility aliases kept for older code paths / logs.
pair_phase_error_max_by_mode = pair_front_mismatch_max_by_mode
front_pair_sync_min_by_mode = (0.0, 0.0, 0.86)
hind_pair_sync_min_by_mode = (0.0, 0.0, 0.82)
lateral_2contact_fraction_max_by_mode = (0.0, 0.0, 0.12)
hind_to_front_lag_steps_by_mode = ((0, 0), (0, 0), (3, 6))
all4_fraction_min_by_mode = (0.0, 0.0, 0.18)
pair2_fraction_min_by_mode = (0.0, 0.0, 0.88)

# Optional foot-clearance predicate. The current BarkourEnv does not yet feed
# clearance_history, so reward_step automatically disables this term when absent.
clearance_min_by_mode = (0.020, 0.025, 0.030)

# Walk mode helper: require occasional >=3-contact support in the recent window.
K_REQUIRE_3PLUS = 11

# -----------------------------------------------------------------------------
# Legacy proxy tolerances retained for compatibility with older code paths.
# -----------------------------------------------------------------------------
eps_diag_sync = 0.20
eps_pair_sync = 0.20
eps_bound_overlap = 0.40
eps_walk_no_flight = 0.0

# -----------------------------------------------------------------------------
# Margin scales for normalized robustness
# -----------------------------------------------------------------------------
# Safety
tau_margin_scale = 5.0
min_contacts_margin_scale = 1.0
com_z_margin_scale = 0.03
abs_vz_margin_scale = 0.10
roll_margin_scale_deg = 5.0
pitch_margin_scale_deg = 5.0
slip_margin_scale = 0.20
support_margin_scale = 0.06

# Final command-relative tracking treatment. Forward error uses a 0.10 m/s
# floor; lateral and yaw use the universal 0.05 scales.
velocity_command_floor = 0.10
track_axis_weights = (1.0, 0.01, 0.01)

# Timing / pattern
stride_margin_scale_by_mode = (0.05, 0.04, 0.04)
duty_margin_scale_by_mode = (0.05, 0.03, 0.05)
diag_phase_margin_scale_by_mode = (0.05, 0.05, 1.00)
diag2_margin_scale_by_mode = (0.05, 0.03, 0.05)
contact2_margin_scale_by_mode = (1.00, 1.00, 1.00)
pair_mismatch_margin_scale_by_mode = (1.00, 1.00, 0.05)
hindfront_margin_scale_by_mode = (1.00, 1.00, 0.10)
flight_margin_scale_by_mode = (1.00, 1.00, 0.05)
front_only_margin_scale_by_mode = (1.00, 1.00, 0.05)
hind_only_margin_scale_by_mode = (1.00, 1.00, 0.05)
all4_margin_scale_by_mode = (1.00, 1.00, 0.08)
event3plus_margin_scale = 1.0
bound_event_margin_scale = 1.0
clearance_margin_scale_by_mode = (0.010, 0.010, 0.010)

# Compatibility alias.
pair_phase_margin_scale_by_mode = pair_mismatch_margin_scale_by_mode

# -----------------------------------------------------------------------------
# Grouped reward weights and tanh alphas
# -----------------------------------------------------------------------------
w_safe_by_mode = (1.0, 1.0, 1.0)
w_track_by_mode = (1.0, 1.0, 1.15)
#w_timing_by_mode = (0.0, 0.0, 1.2)
w_pattern_by_mode = (1.1, 1.2, 1.3)

alpha_safe_by_mode = (0.9, 0.6, 0.8)
alpha_track_by_mode = (0.8, 0.7, 0.8)
#alpha_timing_by_mode = (0.0, 0.0, 0.7)
alpha_pattern_by_mode = (1.2, 0.4, 0.7)

# Torque effort regularizer coefficient
gamma_tau = 1e-6  # 1e-5


# =============================================================================
# Parameters for the STL templates selected by GPT-5.5
# =============================================================================
#
# This section intentionally does not turn any other coefficient above into an
# STL predicate.  It only maps the thresholds required by the symbolic
# templates in artifacts/stl_specs_symbolic.json.

DEG_TO_RAD = 3.141592653589793 / 180.0

# Shared templates.  GPT-5.5 produced a single threshold rather than a
# mode-conditioned one, so <= predicates use the largest configured limit and
# >= predicates use the smallest configured limit.
theta_roll_max = max(roll_abs_by_mode) * DEG_TO_RAD
theta_pitch_max = max(pitch_abs_by_mode) * DEG_TO_RAD
h_base_min = min(com_z_by_mode)
h_base_max = None  # No upper base-height/COM-height limit is present above.
slip_max = max(slip_speed_by_mode)

# The final treatment uses the lateral/yaw entries directly and replaces only
# the forward denominator with max(abs(command_vx), velocity_command_floor).
# The scalar is the weighted sum in velocity_tracking.py; e_vel_max remains
# 1.0, so standing under a nonzero straight-ahead command lies exactly on the
# atomic boundary.
velocity_error_scales = (
    max(eps_vx_by_mode),
    max(eps_vy_by_mode),
    max(eps_yaw_by_mode),
)
e_vel_max = 1.0

# Walk-like templates.
c_walk_min = min_contacts_by_mode[MODE_WALK]
p_3plus_walk_min = 1.0 / K_REQUIRE_3PLUS
p_flight_walk_max = eps_walk_no_flight
duty_walk_min = duty_factor_by_mode[MODE_WALK][0]
clearance_walk_min = clearance_min_by_mode[MODE_WALK]

# Trot-like templates.  diag_2contact_fraction_min_by_mode is interpreted as
# diagonal purity among exactly-two-contact samples; P_2contact separately
# constrains how often exactly two feet are in contact.
p_diag2_trot_min = diag_2contact_fraction_min_by_mode[MODE_TROT]
p_2contact_trot_min = contact2_fraction_min_by_mode[MODE_TROT]
eps_diag_sync_trot_max = diag_phase_error_by_mode[MODE_TROT]
duty_trot_min, duty_trot_max = duty_factor_by_mode[MODE_TROT]
p_flight_trot_max = None  # Only a trot flight minimum is present above.
clearance_trot_min = clearance_min_by_mode[MODE_TROT]

# Bound-like templates.
eps_front_sync_bound_max = pair_front_mismatch_max_by_mode[MODE_BOUND]
eps_hind_sync_bound_max = pair_hind_mismatch_max_by_mode[MODE_BOUND]
front_hind_phase_bound_min, front_hind_phase_bound_max = (
    hind_to_front_lag_by_mode[MODE_BOUND]
)
front_hind_phase_bound_target = (
    front_hind_phase_bound_min + front_hind_phase_bound_max
) / 2.0
eps_front_hind_phase_bound_max = (
    front_hind_phase_bound_max - front_hind_phase_bound_min
) / 2.0
p_front_pair_bound_min = front_only_fraction_min_by_mode[MODE_BOUND]
p_hind_pair_bound_min = hind_only_fraction_min_by_mode[MODE_BOUND]
p_flight_bound_min = flight_fraction_min_by_mode[MODE_BOUND]
duty_bound_max = duty_factor_by_mode[MODE_BOUND][1]
clearance_bound_min = clearance_min_by_mode[MODE_BOUND]

DISABLED_GPT55_STL_THRESHOLDS = (
    "h_base_max",
    "p_flight_trot_max",
)
