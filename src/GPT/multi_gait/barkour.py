import os
import sys
os.environ.setdefault(
    "MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl"
)

import numpy as np
from typing import Any, Dict, Sequence, List
from etils import epath


import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

from brax import base
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model

from reward_config import get_config, get_stl_config
from stl_reward import (
    compute_early_termination_penalty,
    reward_step,
    validate_stl_params,
)
from coeff_config import (
    H,
    EARLY_TERMINATION_PENALTY,
    ENABLE_BOUND_GAIT,
    MODE_WALK, MODE_TROT, MODE_BOUND,
    WALK_TO_TROT_ENTER, TROT_TO_WALK_EXIT,
    TROT_TO_BOUND_ENTER, BOUND_TO_TROT_EXIT,
    cmd_vx_range,
    cmd_vy_range,
    cmd_yaw_range,
    bound_vx_sample_range,
    regime_sample_probs,
)
from velocity_tracking import command_relative_weighted_tracking_error

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


BARKOUR_ROOT_PATH = epath.Path(
    os.environ.get(
        'BARKOUR_ROOT_PATH', 'mujoco_menagerie/google_barkour_vb'
    )
)
STL_REWARD = True

class BarkourEnv(PipelineEnv):
  """Environment for training the barkour quadruped joystick policy in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      scene_file: str = 'scene_mjx.xml',
      reward_weights=None,
      command_sampling: str = 'walk_trot_only',
      bound_straight_commands: bool = True,
      early_termination_penalty: float = EARLY_TERMINATION_PENALTY,
      **kwargs,
  ):
    if early_termination_penalty < 0:
      raise ValueError("early_termination_penalty must be nonnegative.")
    if not ENABLE_BOUND_GAIT and command_sampling != 'walk_trot_only':
      raise ValueError(
          "Phase 1 is walk/trot only; command_sampling must be "
          "'walk_trot_only'."
      )

    path = BARKOUR_ROOT_PATH / scene_file
    if not path.exists():
      raise FileNotFoundError(
          f"Barkour scene not found at {path}. Set BARKOUR_ROOT_PATH to the "
          "google_barkour_vb model directory."
      )
    sys = mjcf.load(path.as_posix())
    self._dt = 0.02  # this environment is 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})

    # override menagerie params for smoother policy
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    if STL_REWARD:
      self.reward_config = get_stl_config()
      validate_stl_params(self.reward_config.stl)
      self.reward_input = None
      self._H = H
    else:
      self.reward_config = get_config()
      # set custom from kwargs
      for k, v in kwargs.items():
        if k.endswith('_scale'):
          self.reward_config.rewards.scales[k[:-6]] = v

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso'
    )
    if reward_weights is None:
        self.reward_weights = None
    else:
      short_keys = ("w_safe", "alpha_safe")
      full_keys = (
          "w_safe", "alpha_safe", "w_track", "alpha_track", "w_gait",
          "alpha_gait",
      )
      if all(k in reward_weights for k in full_keys):
        self.reward_weights = jp.array(
            [reward_weights[k] for k in full_keys], dtype=jp.float32
        )
      elif all(k in reward_weights for k in short_keys):
        self.reward_weights = jp.array(
            [reward_weights[k] for k in short_keys], dtype=jp.float32
        )
      else:
        raise ValueError(
            "reward_weights must provide either the two safety keys or all "
            "six grouped STL keys."
        )

    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._early_termination_penalty = jp.asarray(
        early_termination_penalty, dtype=jp.float32
    )
    self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
    self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
    self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
    self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
    feet_site = [
        'foot_front_left',
        'foot_hind_left',
        'foot_front_right',
        'foot_hind_right',
    ]
    feet_site_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
    self._feet_site_id = np.array(feet_site_id)
    lower_leg_body = [
        'lower_leg_front_left',
        'lower_leg_hind_left',
        'lower_leg_front_right',
        'lower_leg_hind_right',
    ]
    lower_leg_body_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
        for l in lower_leg_body
    ]
    assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
    self._lower_leg_body_id = np.array(lower_leg_body_id)
    self._foot_radius = 0.0175
    self._nv = sys.nv
    self._command_sampling = command_sampling
    self._bound_straight_commands = bound_straight_commands



  def _update_mode_hysteresis(self, mode: jax.Array, command: jax.Array) -> jax.Array:
    vx = jp.abs(command[0])

    mode = jp.where((mode == MODE_WALK) & (vx >= WALK_TO_TROT_ENTER), jp.array(MODE_TROT, dtype=jp.int32), mode)
    mode = jp.where((mode == MODE_TROT) & (vx <= TROT_TO_WALK_EXIT),  jp.array(MODE_WALK, dtype=jp.int32), mode)

    if ENABLE_BOUND_GAIT:
      mode = jp.where((mode == MODE_TROT) & (vx >= TROT_TO_BOUND_ENTER), jp.array(MODE_BOUND, dtype=jp.int32), mode)
      mode = jp.where((mode == MODE_BOUND) & (vx <= BOUND_TO_TROT_EXIT), jp.array(MODE_TROT, dtype=jp.int32), mode)
    else:
      mode = jp.minimum(mode, jp.array(MODE_TROT, dtype=jp.int32))
    return mode  # self.mode = new_mode

  def _mode_from_command(self, command: jax.Array) -> jax.Array:
    """Fresh mode assignment from command speed without hysteresis memory."""
    vx = jp.abs(command[0])
    if not ENABLE_BOUND_GAIT:
      return jp.where(
          vx >= WALK_TO_TROT_ENTER,
          jp.array(MODE_TROT, dtype=jp.int32),
          jp.array(MODE_WALK, dtype=jp.int32),
      )
    return jp.where(
        vx >= TROT_TO_BOUND_ENTER,
        jp.array(MODE_BOUND, dtype=jp.int32),
        jp.where(
            vx >= WALK_TO_TROT_ENTER,
            jp.array(MODE_TROT, dtype=jp.int32),
            jp.array(MODE_WALK, dtype=jp.int32),
        ),
    )

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng, key_regime, key_vx, key_vy, key_yaw = jax.random.split(rng, 5)

    probs = jp.array(regime_sample_probs, dtype=jp.float32)
    probs = probs / jp.sum(probs)

    if self._command_sampling == 'bound_only':
      sampled_regime = jp.array(MODE_BOUND, dtype=jp.int32)
    elif self._command_sampling == 'walk_trot_only':
      walk_trot_probs = probs[:2] / jp.sum(probs[:2])
      sampled_regime = jax.random.choice(
          key_regime,
          jp.array([MODE_WALK, MODE_TROT], dtype=jp.int32),
          shape=(),
          p=walk_trot_probs,
      )
    elif self._command_sampling == 'mixed':
      sampled_regime = jax.random.choice(
          key_regime,
          jp.array([MODE_WALK, MODE_TROT, MODE_BOUND], dtype=jp.int32),
          shape=(),
          p=probs,
      )
    else:
      raise ValueError(
          "command_sampling must be 'walk_trot_only', 'mixed', or "
          "'bound_only'."
      )

    vx_min_global, vx_max_global = cmd_vx_range

    walk_lo = vx_min_global
    walk_hi = jp.maximum(vx_min_global + 1e-6, TROT_TO_WALK_EXIT)

    trot_lo = WALK_TO_TROT_ENTER
    trot_hi = jp.maximum(trot_lo + 1e-6, BOUND_TO_TROT_EXIT)

    bound_lo, bound_hi = bound_vx_sample_range
    bound_hi = jp.minimum(bound_hi, vx_max_global)
    bound_hi = jp.maximum(bound_lo + 1e-6, bound_hi)

    def _sample_uniform(key, lo, hi):
      return jax.random.uniform(key, shape=(), minval=lo, maxval=hi)

    vx_walk = _sample_uniform(key_vx, walk_lo, walk_hi)
    vx_trot = _sample_uniform(key_vx, trot_lo, trot_hi)
    vx_bound = _sample_uniform(key_vx, bound_lo, bound_hi)

    vx = jp.where(
        sampled_regime == MODE_WALK,
        vx_walk,
        jp.where(sampled_regime == MODE_TROT, vx_trot, vx_bound),
    )

    sampled_vy = jax.random.uniform(
        key_vy, shape=(), minval=cmd_vy_range[0], maxval=cmd_vy_range[1]
    )
    sampled_yaw = jax.random.uniform(
        key_yaw, shape=(), minval=cmd_yaw_range[0], maxval=cmd_yaw_range[1]
    )

    vy = jp.where(
        (sampled_regime == MODE_BOUND) & self._bound_straight_commands,
        0.0,
        sampled_vy,
    )
    yaw = jp.where(
        (sampled_regime == MODE_BOUND) & self._bound_straight_commands,
        0.0,
        sampled_yaw,
    )

    return jp.array([vx, vy, yaw], dtype=jp.float32)


  def reset(self, rng: jax.Array) -> State:
    rng, key = jax.random.split(rng)

    pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
    command = self.sample_command(key)
    mode0 = self._mode_from_command(command)

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'command': command,
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
        'kick': jp.array([0.0, 0.0]),
        'step': 0,

        'history_len': jp.array(0, dtype=jp.int32),
        'mode': mode0,

        'tau_history': jp.zeros((self._H, 12)),
        'roll_history': jp.zeros((self._H,)),
        'pitch_history': jp.zeros((self._H,)),
        'base_height_history': jp.zeros((self._H,)),
        'velocity_tracking_error_history': jp.zeros((self._H,)),
        'contact_history': jp.zeros((self._H, 4)),
        'foot_clearance_history': jp.zeros((self._H, 4)),
        'foot_slip_history': jp.zeros((self._H, 4)),
        'action_delta_history': jp.zeros((self._H,)),
    }

    obs_history = jp.zeros(15 * 34) # jp.zeros(15 * 32)  # store 15 steps of history
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

    # kick
    push_interval = 10
    kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    kick *= jp.mod(state.info['step'], push_interval) == 0
    qvel = state.pipeline_state.qvel
    qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
    state = state.tree_replace({'pipeline_state.qvel': qvel})

    # physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    # observation data
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    # foot contact data based on z-position with Schmitt-trigger hysteresis - disabled
    foot_pos = pipeline_state.site_xpos[self._feet_site_id]
    foot_contact_z = foot_pos[:, 2] - self._foot_radius
    #contact_on = foot_contact_z < self._contact_z_on
    #contact_off = foot_contact_z > self._contact_z_off
    contact = foot_contact_z < 1e-3  # a mm or less off the floor
    contact_filt_mm = contact | state.info['last_contact']
    contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
    state.info['feet_air_time'] += self.dt


    # done if joint limits are reached or robot is falling
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18


    history_len = jp.minimum(state.info['history_len'] + 1, self._H)
    mode = self._update_mode_hysteresis(state.info['mode'], state.info['command'])

    # STL signal extraction and history update.
    torso_i = self._torso_idx - 1
    roll_abs, pitch_abs = self._base_roll_pitch_abs(x.rot[torso_i])
    base_height = x.pos[torso_i, 2]
    velocity_tracking_error = self._velocity_tracking_error_scalar(
        state.info['command'], x, xd, torso_i
    )

    # qfrc_actuator may include floating-base entries; only the 12 motors are
    # relevant to GPT-5.5's Joint_torque predicate.
    actuated_tau = pipeline_state.qfrc_actuator[-12:]
    contact_float = contact_filt_mm.astype(jp.float32)
    foot_clearance = jp.maximum(foot_contact_z, 0.0)
    foot_vel = self._foot_velocities(pipeline_state, foot_pos)
    foot_slip = (
        jp.linalg.norm(foot_vel[:, :2], axis=1)
        * contact_filt_cm.astype(jp.float32)
    )
    action_delta = jp.sum(jp.square(action - state.info['last_act']))

    tau_history = self._append_history(state.info['tau_history'], actuated_tau)
    roll_history = self._append_history(state.info['roll_history'], roll_abs)
    pitch_history = self._append_history(state.info['pitch_history'], pitch_abs)
    base_height_history = self._append_history(
        state.info['base_height_history'], base_height
    )
    velocity_tracking_error_history = self._append_history(
        state.info['velocity_tracking_error_history'], velocity_tracking_error
    )
    contact_history = self._append_history(
        state.info['contact_history'], contact_float
    )
    foot_clearance_history = self._append_history(
        state.info['foot_clearance_history'], foot_clearance
    )
    foot_slip_history = self._append_history(
        state.info['foot_slip_history'], foot_slip
    )
    action_delta_history = self._append_history(
        state.info['action_delta_history'], action_delta
    )

    if STL_REWARD:
      reward_input = {
          'tau_history': tau_history,
          'roll_history': roll_history,
          'pitch_history': pitch_history,
          'base_height_history': base_height_history,
          'velocity_tracking_error_history': velocity_tracking_error_history,
          'contact_history': contact_history,
          'foot_clearance_history': foot_clearance_history,
          'foot_slip_history': foot_slip_history,
          'action_delta_history': action_delta_history,
      }

      (
          r,
          tau_effort,
          action_rate_effort,
          rho_safety,
          rho_tracking,
          rho_gait,
          rho_safety_raw,
          rho_tracking_raw,
          rho_gait_raw,
          rho_torque,
      ) = reward_step(
          reward_input=reward_input,
          commands=state.info['command'],
          mode=mode,
          valid_len=history_len,
          stl_params=self.reward_config.stl,
          weights_override=self.reward_weights,
      )

      rewards = {
          'total_stl_reward': r,
          'rho_safety': rho_safety,
          'rho_tracking': rho_tracking,
          'rho_gait': rho_gait,
          'rho_safety_raw': rho_safety_raw,
          'rho_tracking_raw': rho_tracking_raw,
          'rho_gait_raw': rho_gait_raw,
          'torque_lim': rho_torque,
          'torque_effort': tau_effort,
          'smooth_action': action_rate_effort,
      }

      termination_penalty = compute_early_termination_penalty(
          done, self._early_termination_penalty
      )
      rewards['early_termination_penalty'] = termination_penalty
      reward = jp.clip(
          r * self.dt + termination_penalty, -100.0, 1000.0
      )

    else:
      rewards = {
          'tracking_lin_vel': (
              self._reward_tracking_lin_vel(state.info['command'], x, xd)
          ),
          'tracking_ang_vel': (
              self._reward_tracking_ang_vel(state.info['command'], x, xd)
          ),
          'lin_vel_z': self._reward_lin_vel_z(xd),
          'ang_vel_xy': self._reward_ang_vel_xy(xd),
          'orientation': self._reward_orientation(x),
          'torques': self._reward_torques(pipeline_state.qfrc_actuator),
          'action_rate': self._reward_action_rate(action, state.info['last_act']),
          'stand_still': self._reward_stand_still(
              state.info['command'], joint_angles,
          ),
          'feet_air_time': self._reward_feet_air_time(
              state.info['feet_air_time'],
              first_contact,
              state.info['command'],
          ),
          'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
          'termination': self._reward_termination(done, state.info['step']),
      }
      rewards = {
          k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
      }
      reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # state management
    state.info['kick'] = kick
    state.info['last_act'] = action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt_mm
    state.info['last_contact'] = contact
    state.info['rewards'] = rewards
    state.info['step'] += 1
    state.info['rng'] = rng

    state.info['tau_history'] = tau_history
    state.info['roll_history'] = roll_history
    state.info['pitch_history'] = pitch_history
    state.info['base_height_history'] = base_height_history
    state.info['velocity_tracking_error_history'] = velocity_tracking_error_history
    state.info['contact_history'] = contact_history
    state.info['foot_clearance_history'] = foot_clearance_history
    state.info['foot_slip_history'] = foot_slip_history
    state.info['action_delta_history'] = action_delta_history

    state.info['history_len'] = history_len
    state.info['mode'] = mode

    done_bool = done
    command_reset = done_bool | (state.info['step'] > 500)

    # On either timeout or termination, start a fresh command and clear all
    # command-conditioned histories so the next [t-H, t] window is consistent.
    sampled_command = self.sample_command(cmd_rng)
    next_command = jp.where(command_reset, sampled_command, state.info['command'])
    next_mode = jp.where(
        command_reset,
        self._mode_from_command(next_command),
        state.info['mode'],
    )

    state.info['command'] = next_command
    state.info['step'] = jp.where(command_reset, 0, state.info['step'])


    # This makes the [t-H, t] reward window consistent with the current command.
    def _maybe_clear(buf):
      return jp.where(command_reset, jp.zeros_like(buf), buf)

    for hist_key in (
        'tau_history',
        'roll_history',
        'pitch_history',
        'base_height_history',
        'velocity_tracking_error_history',
        'contact_history',
        'foot_clearance_history',
        'foot_slip_history',
        'action_delta_history',
    ):
      state.info[hist_key] = _maybe_clear(state.info[hist_key])

    state.info['history_len'] = jp.where(command_reset, jp.array(0, dtype=jp.int32), state.info['history_len'])
    state.info['mode'] = next_mode

    # Also clear action / velocity history terms that feed the observation stack.
    state.info['last_act'] = jp.where(command_reset, jp.zeros_like(state.info['last_act']), state.info['last_act'])
    state.info['last_vel'] = jp.where(command_reset, jp.zeros_like(state.info['last_vel']), state.info['last_vel'])
    state.info['last_contact'] = jp.where(command_reset, jp.zeros_like(state.info['last_contact']), state.info['last_contact'])
    state.info['feet_air_time'] = jp.where(command_reset, jp.zeros_like(state.info['feet_air_time']), state.info['feet_air_time'])
    state.info['kick'] = jp.where(command_reset, jp.zeros_like(state.info['kick']), state.info['kick'])

    obs = jp.where(command_reset, self._get_obs(pipeline_state, state.info, jp.zeros_like(state.obs)), obs)

    # log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]  # or state.metrics['total_dist'] = x.pos[self._torso_idx - 1, 0]
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done_bool)
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

    mode_one_hot = jax.nn.one_hot(state_info['mode'], 3, dtype=jp.float32)  # [walk, trot, bound]

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        pipeline_state.q[7:] - self._default_pose,           # motor angles
        state_info['last_act'],                              # last action
        mode_one_hot,                                      # jp.array([state_info['mode']], dtype=jp.float32),
    ])

    # clip, noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    # stack observations through time

    # obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
    obs = jp.roll(obs_history, obs.shape[0]).at[:obs.shape[0]].set(obs)

    return obs

  # ------------ reward functions----------------
  def _append_history(self, buf: jax.Array, value: jax.Array) -> jax.Array:
    return jp.roll(buf, -1, axis=0).at[-1].set(value)

  def _base_roll_pitch_abs(self, quat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns absolute roll and pitch in radians from a wxyz quaternion."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = jp.arcsin(jp.clip(sinp, -1.0, 1.0))
    return jp.abs(roll), jp.abs(pitch)

  def _velocity_tracking_error_scalar(
      self,
      commands: jax.Array,
      x: Transform,
      xd: Motion,
      torso_index: int,
  ) -> jax.Array:
    """Returns the final command-relative weighted tracking error."""
    inv_torso_rot = math.quat_inv(x.rot[torso_index])
    local_vel = math.rotate(xd.vel[torso_index], inv_torso_rot)
    local_ang_vel = math.rotate(xd.ang[torso_index], inv_torso_rot)
    return command_relative_weighted_tracking_error(
        command=commands,
        body_linear_velocity=local_vel,
        body_yaw_rate=local_ang_vel[2],
    ).velocity_error

  def _foot_velocities(
      self,
      pipeline_state: base.State,
      foot_pos: jax.Array,
  ) -> jax.Array:
    """Returns Cartesian velocities at the four foot sites."""
    feet_offset = foot_pos - pipeline_state.xpos[self._lower_leg_body_id]
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1
    return offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.square(xd.vel[0, 2])

  def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(xd.ang[0, :2]))

  def _reward_orientation(self, x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(
        -lin_vel_error / self.reward_config.rewards.tracking_sigma
    )
    return lin_vel_reward

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= (
        math.normalize(commands[:2])[1] > 0.05
    )  # no reward for zero command
    return rew_air_time

  def _reward_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _reward_foot_slip(
      self, pipeline_state: base.State, contact_filt: jax.Array
  ) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

  def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera, width=width, height=height)

#envs.register_environment('barkour', BarkourEnv)

# %%
