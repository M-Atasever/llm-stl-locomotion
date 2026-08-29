#%%
import os
import sys
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
from stl_reward import reward_step
from coeff_config import (
    EARLY_TERMINATION_PENALTY,
    H,
    MODE_WALK, MODE_TROT, MODE_BOUND,
    TRAINING_COMMAND_SAMPLING,
    WALK_TO_TROT_ENTER, TROT_TO_WALK_EXIT,
    TROT_TO_BOUND_ENTER, BOUND_TO_TROT_EXIT,
    cmd_vx_range,
    cmd_vy_range,
    cmd_yaw_range,
    bound_vx_sample_range,
    regime_sample_probs,
)

os.environ.setdefault(
    'MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'disable'
)

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


_DEFAULT_BARKOUR_ROOT_PATH = (
    epath.Path(__file__).parent
    / 'mujoco_menagerie'
    / 'google_barkour_vb'
)
BARKOUR_ROOT_PATH = epath.Path(
    os.environ.get('BARKOUR_ROOT_PATH', _DEFAULT_BARKOUR_ROOT_PATH.as_posix())
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
      command_sampling: str = TRAINING_COMMAND_SAMPLING,
      bound_straight_commands: bool = True,
      early_termination_penalty: float = EARLY_TERMINATION_PENALTY,
      curriculum_vx_range=None,
      **kwargs,
  ):
    if command_sampling not in (
        'walk_trot_only', 'mixed', 'bound_only', 'curriculum'
    ):
      raise ValueError(
          'command_sampling must be walk_trot_only, mixed, bound_only, or '
          f'curriculum; got {command_sampling!r}'
      )
    if command_sampling == 'curriculum':
      if curriculum_vx_range is None:
        raise ValueError(
            'curriculum command sampling requires curriculum_vx_range.'
        )
      lo, hi = float(curriculum_vx_range[0]), float(curriculum_vx_range[1])
      if not 0.0 <= lo < hi:
        raise ValueError(
            f'curriculum_vx_range must satisfy 0 <= lo < hi; got ({lo}, {hi})'
        )
      if hi > cmd_vx_range[1] + 1e-9:
        raise ValueError(
            f'curriculum_vx_range upper bound {hi} exceeds the global command '
            f'range cap {cmd_vx_range[1]}.'
        )
      curriculum_vx_range = (lo, hi)
    elif curriculum_vx_range is not None:
      raise ValueError(
          'curriculum_vx_range is only valid with command_sampling='
          "'curriculum'."
      )
    if early_termination_penalty < 0:
      raise ValueError('early_termination_penalty must be nonnegative.')
    path = BARKOUR_ROOT_PATH / scene_file
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
    if reward_weights is not None:
      raise ValueError(
          "Custom subgroup reward weights are disabled because the STL reward "
          "is the smooth robustness of the active gait-conditioned formula."
      )
    self.reward_weights = None

    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
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
    self._curriculum_vx_range = curriculum_vx_range
    self._bound_straight_commands = bound_straight_commands
    self._early_termination_penalty = jp.asarray(
        early_termination_penalty, dtype=jp.float32
    )



  def _update_mode_hysteresis(self, mode: jax.Array, command: jax.Array) -> jax.Array:
    vx = jp.abs(command[0])

    mode = jp.where((mode == MODE_WALK) & (vx >= WALK_TO_TROT_ENTER), jp.array(MODE_TROT, dtype=jp.int32), mode)
    mode = jp.where((mode == MODE_TROT) & (vx <= TROT_TO_WALK_EXIT),  jp.array(MODE_WALK, dtype=jp.int32), mode)

    mode = jp.where((mode == MODE_TROT) & (vx >= TROT_TO_BOUND_ENTER), jp.array(MODE_BOUND, dtype=jp.int32), mode)
    mode = jp.where((mode == MODE_BOUND) & (vx <= BOUND_TO_TROT_EXIT), jp.array(MODE_TROT, dtype=jp.int32), mode)
    return mode  # self.mode = new_mode

  def _mode_from_command(self, command: jax.Array) -> jax.Array:
    """Fresh mode assignment from command speed without hysteresis memory."""
    vx = jp.abs(command[0])
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

    if self._command_sampling == 'curriculum':
      # Curriculum sampling is gait-stratified inside the active stage range.
      # Reachable regimes use the configured (walk, trot, bound) probabilities,
      # renormalized when a stage has not introduced a faster gait yet.  This
      # makes bound an intentional 40% of stages 7-8 instead of the small tail
      # produced by one uniform draw over the complete 0..vx_max interval.
      lo, hi = self._curriculum_vx_range
      reachable = jp.array(
          (
              True,
              hi > WALK_TO_TROT_ENTER,
              hi > TROT_TO_BOUND_ENTER,
          ),
          dtype=bool,
      )
      probs = jp.where(
          reachable,
          jp.array(regime_sample_probs, dtype=jp.float32),
          0.0,
      )
      probs = probs / jp.sum(probs)
      sampled_regime = jax.random.choice(
          key_regime,
          jp.array([MODE_WALK, MODE_TROT, MODE_BOUND], dtype=jp.int32),
          shape=(),
          p=probs,
      )

      walk_lo = lo
      walk_hi = jp.maximum(walk_lo + 1e-6, jp.minimum(hi, WALK_TO_TROT_ENTER))
      trot_lo = jp.maximum(lo, WALK_TO_TROT_ENTER)
      trot_hi = jp.maximum(trot_lo + 1e-6, jp.minimum(hi, TROT_TO_BOUND_ENTER))
      bound_lo = jp.maximum(lo, TROT_TO_BOUND_ENTER)
      bound_hi = jp.maximum(
          bound_lo + 1e-6,
          jp.minimum(hi, bound_vx_sample_range[1]),
      )

      vx_walk = jax.random.uniform(
          key_vx, shape=(), minval=walk_lo, maxval=walk_hi
      )
      vx_trot = jax.random.uniform(
          key_vx, shape=(), minval=trot_lo, maxval=trot_hi
      )
      vx_bound = jax.random.uniform(
          key_vx, shape=(), minval=bound_lo, maxval=bound_hi
      )
      vx = jp.where(
          sampled_regime == MODE_WALK,
          vx_walk,
          jp.where(sampled_regime == MODE_TROT, vx_trot, vx_bound),
      )
    else:
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
      else:
        sampled_regime = jax.random.choice(
            key_regime,
            jp.array([MODE_WALK, MODE_TROT, MODE_BOUND], dtype=jp.int32),
            shape=(),
            p=probs,
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

        'tau_history': jp.zeros((self._H, 18)),
        # Columns:
        # 0 e_vx
        # 1 e_vy
        # 2 e_wz
        # 3 center-of-mass z-coordinate
        # 4 max(abs(roll), abs(pitch))
        # 5 n_stance_contacts
        'stl_history': jp.zeros((self._H, 6), dtype=jp.float32),
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


    tau_history = (
        jp.roll(state.info['tau_history'], -1, axis=0)
        .at[-1]
        .set(pipeline_state.qfrc_actuator)
    )

    stl_signal = self._get_stl_signals(
        pipeline_state=pipeline_state,
        command=state.info['command'],
        contact=contact,
    )

    stl_history = (
        jp.roll(state.info['stl_history'], -1, axis=0)
        .at[-1]
        .set(stl_signal)
    )

    history_len = jp.minimum(state.info['history_len'] + 1, self._H)
    mode = self._update_mode_hysteresis(state.info['mode'], state.info['command'])

    # calculate reward

    if STL_REWARD:

      reward_input = {
          'stl_history': stl_history,
          'tau_history': tau_history,
      }

      (
          r,
          rho_stl,
          rho_stl_smooth,
          rho_tracking,
          rho_safety,
          rho_contact,
          rho_com_z,
          rho_tilt,
          rho_torque,
          tau_effort,
      ) = reward_step(
          reward_input=reward_input,
          commands=state.info['command'],
          mode=mode,
          valid_len=history_len,
          weights_override=self.reward_weights,
      )

      rewards = {
          'total_stl_reward': r,
          'rho_stl': rho_stl,
          'rho_stl_smooth': rho_stl_smooth,
          'rho_tracking': rho_tracking,
          'rho_safety': rho_safety,
          'rho_contact': rho_contact,
          'rho_com_z': rho_com_z,
          'rho_tilt': rho_tilt,

          # Optional non-STL diagnostics.
          'torque_lim': rho_torque,
          'smooth_action': tau_effort,

          # PPO survival safeguard; this is not an STL predicate.
          'early_termination_penalty': (
              -self._early_termination_penalty * done.astype(jp.float32)
          ),
      }

      reward = jp.clip(
          r * self.dt + rewards['early_termination_penalty'],
          -100.0,
          1000.0,
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
    state.info['stl_history'] = stl_history

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

    state.info['tau_history'] = _maybe_clear(state.info['tau_history'])
    state.info['stl_history'] = _maybe_clear(state.info['stl_history'])

    state.info['history_len'] = jp.where(
        command_reset,
        jp.array(0, dtype=jp.int32),
        state.info['history_len'],
    )
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

  def _get_stl_signals(
      self,
      pipeline_state: base.State,
      command: jax.Array,
      contact: jax.Array,
  ) -> jax.Array:
    """Extracts the scalar signals used by the STL reward.

    Returns:
      array shape (6,):
        0: e_vx = |v_x - v_x_cmd|
        1: e_vy = |v_y - v_y_cmd|
        2: e_wz = |omega_z - omega_z_cmd|
        3: center-of-mass z-coordinate
        4: max(abs(roll), abs(pitch))
        5: n_stance_contacts
    """

    torso_i = self._torso_idx - 1

    x = pipeline_state.x
    xd = pipeline_state.xd

    torso_rot = x.rot[torso_i]
    inv_torso_rot = math.quat_inv(torso_rot)

    local_lin_vel = math.rotate(xd.vel[torso_i], inv_torso_rot)
    local_ang_vel = math.rotate(xd.ang[torso_i], inv_torso_rot)

    e_vx = jp.abs(local_lin_vel[0] - command[0])
    e_vy = jp.abs(local_lin_vel[1] - command[1])
    e_wz = jp.abs(local_ang_vel[2] - command[2])

    com_z = pipeline_state.subtree_com[0, 2]

    gravity_body = math.rotate(
        jp.array([0.0, 0.0, -1.0]),
        inv_torso_rot,
    )
    roll = jp.atan2(gravity_body[1], -gravity_body[2])
    pitch = jp.atan2(
        -gravity_body[0],
        jp.sqrt(gravity_body[1] ** 2 + gravity_body[2] ** 2),
    )
    body_tilt_angle = jp.maximum(jp.abs(roll), jp.abs(pitch))

    n_stance_contacts = jp.sum(contact.astype(jp.float32))

    return jp.array(
        [
            e_vx,
            e_vy,
            e_wz,
            com_z,
            body_tilt_angle,
            n_stance_contacts,
        ],
        dtype=jp.float32,
    )

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
