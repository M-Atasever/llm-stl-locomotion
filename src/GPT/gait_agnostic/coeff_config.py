# =========================
# Barkour STL reward config
# =========================

# -----------------------------------------------------------------------------
# Actuator / torque settings
# -----------------------------------------------------------------------------

# Barkour vb joint torque limit.
# This is not part of the JSON STL spec, but is kept as a regularizer/diagnostic.
tau_max = 18.0  # Nm

# Torque effort regularizer coefficient.
gamma_tau = 1e-6

# -----------------------------------------------------------------------------
# Temporal settings
# -----------------------------------------------------------------------------

DT = 0.02  # 50 Hz control step

# Final selected STL horizon. H = 1 corresponds to one 0.02-second control
# sample at 50 Hz.
H = 1
H_by_mode = H  # Compatibility alias for the supplied project code.

# During the first few steps after reset/command-change, the STL history is not
# full. We evaluate the STL formula over the available suffix of the history.
H_WARMUP_MIN_VALID = 1

# -----------------------------------------------------------------------------
# Gait mode IDs
# -----------------------------------------------------------------------------

MODE_WALK = 0
MODE_TROT = 1
MODE_BOUND = 2

# The final pipeline supports all three command regimes. The curriculum trainer
# passes command_sampling="curriculum" per stage; the mixed default keeps bound
# enabled for tools that instantiate BarkourEnv directly.
ENABLE_BOUND_GAIT = True
TRAINING_GAIT_MODES = (MODE_WALK, MODE_TROT, MODE_BOUND)
TRAINING_COMMAND_SAMPLING = "mixed"
TRAINING_NUM_TIMESTEPS = 400_000_000
TRAINING_NUM_ENVS = 8192
# The first five speeds exercise walk/trot; the final two exercise bound.
VISUAL_EVAL_SPEEDS = (0.3, 0.5, 0.9, 1.0, 1.2, 1.7, 1.9)

# -----------------------------------------------------------------------------
# Eight-stage velocity curriculum
# -----------------------------------------------------------------------------
# Each stage first samples a reachable gait using the configured regime
# probabilities, then samples forward velocity uniformly inside that gait's
# portion of the stage range. The probabilities are renormalized until faster
# gaits become reachable (walk < 0.72 <= trot < 1.65 <= bound). Stages 1-6
# therefore stay within walk/trot; stages 7-8 intentionally sample bound.
CURRICULUM_STAGES = (
    {"stage": 1, "vx_range": (0.0, 0.5), "num_timesteps": 50_000_000},
    {"stage": 2, "vx_range": (0.0, 0.75), "num_timesteps": 50_000_000},
    {"stage": 3, "vx_range": (0.0, 1.0), "num_timesteps": 50_000_000},
    {"stage": 4, "vx_range": (0.0, 1.2), "num_timesteps": 50_000_000},
    {"stage": 5, "vx_range": (0.0, 1.4), "num_timesteps": 50_000_000},
    {"stage": 6, "vx_range": (0.0, 1.6), "num_timesteps": 50_000_000},
    {"stage": 7, "vx_range": (0.0, 1.8), "num_timesteps": 50_000_000},
    {"stage": 8, "vx_range": (0.0, 1.9), "num_timesteps": 50_000_000},
)
CURRICULUM_GAIT_MODES = (MODE_WALK, MODE_TROT, MODE_BOUND)
CURRICULUM_INCLUDES_BOUND = True
CURRICULUM_USES_REGIME_SAMPLE_PROBS = True

# -----------------------------------------------------------------------------
# Data-driven predicate gate (50 expert trajectories x timesteps 50..499)
# -----------------------------------------------------------------------------

Q50_FILTER_DROP_PREFIX_TIMESTEPS = 50
Q50_FILTER_OBSERVATIONS_PER_GAIT = 22_500
# All three expert datasets are rescored under these common thresholds before
# q50 selection.  The 0.60 forward tolerance is retained only for comparability
# with the original expert analysis; PPO training still uses eps_vx=0.25 below.
Q50_ANALYSIS_EPS_VX = 0.60
Q50_ANALYSIS_EPS_VY = 0.05
Q50_ANALYSIS_EPS_WZ = 0.05
Q50_ANALYSIS_Z_MIN = 0.16
Q50_ANALYSIS_THETA_MAX_DEGREES = 15.0
Q50_ANALYSIS_N_CONTACT_MIN = 0
STL_PREDICATE_ORDER = (
    "Gpast1_vx_track",
    "Gpast1_vy_track",
    "Gpast1_yaw_rate_track",
    "Gpast1_base_height_lower",
    "Gpast1_body_tilt",
    "Gpast1_contact_count_lower",
)

WALK_Q50_ROBUSTNESS = (
    0.5429699420928955,
    0.007973596453666687,
    -0.0651620477437973,
    0.10048359632492065,
    0.23605899512767792,
    2.0,
)
TROT_Q50_ROBUSTNESS = (
    0.49309277534484863,
    -0.02240310236811638,
    -0.12107747793197632,
    0.0930476188659668,
    0.2259882688522339,
    2.0,
)
BOUND_Q50_ROBUSTNESS = (
    0.1502612829208374,
    -0.02986205741763115,
    -0.20263312757015228,
    0.0580257773399353,
    0.13877367973327637,
    0.0,
)

# The baseline masks preserve the exact outcome of the original rule: a
# predicate passes when its gait-specific expert-data q50 is nonnegative.
# They are retained as provenance and validated independently from the active
# training masks below.
WALK_Q50_BASELINE_PREDICATE_MASK = (
    True,
    True,
    False,
    True,
    True,
    True,
)
TROT_Q50_BASELINE_PREDICATE_MASK = (
    True,
    False,
    False,
    True,
    True,
    True,
)
BOUND_Q50_BASELINE_PREDICATE_MASK = (
    True,
    False,
    False,
    True,
    True,
    True,
)

# Final pre-training overrides. The common q50 analysis excludes yaw for all
# three gaits and lateral tracking for trot and bound. Both predicates remain
# active in the formula, but receive one tenth of the forward-tracking
# importance in the dense smooth reward. Raw q50 values and baseline masks
# remain above for auditability.
LATERAL_PREDICATE_INDEX = 1
YAW_PREDICATE_INDEX = 2
LATERAL_Q50_OVERRIDE_ENABLED = True
YAW_Q50_OVERRIDE_ENABLED = True
LATERAL_Q50_OVERRIDE_REASON = (
    "The final configuration keeps lateral-velocity tracking active with "
    "reduced importance relative to forward tracking."
)
YAW_Q50_OVERRIDE_REASON = (
    "The final configuration keeps yaw-rate tracking active with reduced "
    "importance to limit accumulated heading error."
)
WALK_ACTIVE_PREDICATE_MASK = (True, True, True, True, True, True)
TROT_ACTIVE_PREDICATE_MASK = (True, True, True, True, True, True)
BOUND_ACTIVE_PREDICATE_MASK = (True, True, True, True, True, True)

WALK_EXCLUDED_PREDICATES = ()
TROT_EXCLUDED_PREDICATES = ()
BOUND_EXCLUDED_PREDICATES = ()

# Non-STL survival safeguard. This unscaled cost is applied only on a
# physical-failure transition, preventing a policy from ending an episode to
# avoid future negative robustness. It does not change the STL formula.
EARLY_TERMINATION_PENALTY = 1.6

# -----------------------------------------------------------------------------
# Command-speed hysteresis on |vx| [m/s]
# -----------------------------------------------------------------------------

WALK_TO_TROT_ENTER = 0.72
TROT_TO_WALK_EXIT = 0.65

# Final gait-transition thresholds keep the 0.0-1.6 curriculum stage
# walk/trot-only.
TROT_TO_BOUND_ENTER = 1.65
BOUND_TO_TROT_EXIT = 1.60

# Mixed-regime sampling and bound-focused curriculum helpers.
regime_sample_probs = (0.25, 0.35, 0.40)
cmd_vx_range = (0.0, 1.9)
cmd_vy_range = (-0.2, 0.2)
cmd_yaw_range = (-0.2, 0.2)
bound_vx_sample_range = (1.65, 1.9)

# -----------------------------------------------------------------------------
# STL symbolic parameters selected from the authoritative coefficient config.
# -----------------------------------------------------------------------------
# Velocity tracking tolerances.
# The final forward-error tolerance is 0.25 m/s.
eps_vx = 0.25  # [m/s]
eps_vy = 0.05  # [m/s] max(eps_vy_by_mode)
eps_wz = 0.05  # [rad/s] max(eps_yaw_by_mode)

# Center-of-mass height lower bound.  Bound is now active and the generated
# formula has one universal threshold, so the >= rule uses the minimum across
# walk/trot/bound: min(0.18, 0.22, 0.16) = 0.16 m.
z_min_by_training_mode = (0.18, 0.22, 0.16)
z_min = min(z_min_by_training_mode)  # [m]
# z_max = 0.45  # DISABLED: unsupported upper center-of-mass height bound.

# Body tilt bound for max(abs(roll), abs(pitch)).
# The universal <= rule takes the largest applicable walk/trot/bound roll or
# pitch limit: max(10, 7, 15, 8, 7, 15) = 15 degrees.
roll_abs_deg_by_training_mode = (10.0, 7.0, 15.0)
pitch_abs_deg_by_training_mode = (8.0, 7.0, 15.0)
theta_max_degrees = max(
    roll_abs_deg_by_training_mode + pitch_abs_deg_by_training_mode
)
theta_max = theta_max_degrees * 3.141592653589793 / 180.0

# Stance-contact-count lower bound.  Bound permits flight phases, so the
# universal >= threshold is min(2, 2, 0) = 0 contacts.
min_contacts_by_training_mode = (2, 2, 0)
n_contact_min = min(min_contacts_by_training_mode)
# n_contact_max = 4.5  # DISABLED: unsupported upper contact-count bound.

# -----------------------------------------------------------------------------
# Smooth robustness and config-derived normalization scales
# -----------------------------------------------------------------------------
# beta is the smooth-min sharpness from the authoritative config.
beta = 0.5

# Tracking predicates use their configured tolerances as natural scales.
velocity_vx_margin_scale = eps_vx
velocity_vy_margin_scale = eps_vy
velocity_wz_margin_scale = eps_wz

com_z_margin_scale = 0.03  # [m]
tilt_margin_scale = 5.0 * 3.141592653589793 / 180.0  # [rad]
min_contacts_margin_scale = 1.0  # [feet]

# -----------------------------------------------------------------------------
# Predicate importance in the dense smooth-robustness reward
# -----------------------------------------------------------------------------
# Exact STL robustness and satisfaction checks remain unweighted.  These
# positive prior weights only control each active predicate's influence in the
# differentiable smooth conjunction used as the PPO reward. The final
# configuration gives yaw and lateral tracking one tenth the weight of forward
# tracking.
FORWARD_TRACKING_REWARD_WEIGHT = 1.0
LATERAL_TRACKING_REWARD_WEIGHT = 0.1
YAW_TRACKING_REWARD_WEIGHT = 0.1
TRACKING_AUX_TO_FORWARD_WEIGHT_RATIO = 0.1
PREDICATE_REWARD_WEIGHTS = (
    FORWARD_TRACKING_REWARD_WEIGHT,
    LATERAL_TRACKING_REWARD_WEIGHT,
    YAW_TRACKING_REWARD_WEIGHT,
    1.0,  # center-of-mass height lower bound
    1.0,  # body tilt upper bound
    1.0,  # stance-contact-count lower bound
)
