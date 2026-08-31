#%%
"""Barkour environment with the pre-q50 collection STL signals."""

import os
import sys
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
  os.environ["XLA_FLAGS"] = (xla_flags + " --xla_gpu_triton_gemm_any=True").strip()

import jax
from jax import numpy as jp
import mujoco
import numpy as np
from etils import epath

from brax import base
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

try:
  from . import coeff_config as _coeff
  from .reward_config import get_config, get_stl_config
  from .stl_reward import SIGNAL_KEYS, reward_step
except ImportError:
  import coeff_config as _coeff
  from reward_config import get_config, get_stl_config
  from stl_reward import SIGNAL_KEYS, reward_step

BOUND_TO_TROT_EXIT = _coeff.BOUND_TO_TROT_EXIT
DT = _coeff.DT
H = _coeff.H
MODE_BOUND = _coeff.MODE_BOUND
MODE_TROT = _coeff.MODE_TROT
MODE_WALK = _coeff.MODE_WALK
TROT_TO_BOUND_ENTER = _coeff.TROT_TO_BOUND_ENTER
TROT_TO_WALK_EXIT = _coeff.TROT_TO_WALK_EXIT
WALK_TO_TROT_ENTER = _coeff.WALK_TO_TROT_ENTER
bound_vx_sample_range = _coeff.bound_vx_sample_range
cmd_vx_range = _coeff.cmd_vx_range
cmd_vy_range = _coeff.cmd_vy_range
cmd_yaw_range = _coeff.cmd_yaw_range
regime_sample_probs = _coeff.regime_sample_probs


np.set_printoptions(precision=3, suppress=True, linewidth=100)


BARKOUR_ROOT_PATH = epath.Path(os.environ.get("BARKOUR_ROOT_PATH", "PLACEHOLDER")).expanduser()
STL_REWARD = True


class BarkourEnv(PipelineEnv):
  """Environment for training a Barkour quadruped joystick policy in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      scene_file: str = "scene_mjx.xml",
      reward_weights=None,
      command_sampling: str = "mixed",
      bound_straight_commands: bool = True,
      **kwargs,
  ):
    if str(BARKOUR_ROOT_PATH) == "PLACEHOLDER":
      raise FileNotFoundError(
          "PLACEHOLDER: set BARKOUR_ROOT_PATH to the directory containing scene_mjx.xml."
      )
    path = BARKOUR_ROOT_PATH / scene_file
    if not path.is_file():
      raise FileNotFoundError(f"Barkour scene file not found: {path}")
    sys = mjcf.load(path.as_posix())
    self._dt = DT
    sys = sys.tree_replace({"opt.timestep": 0.004})
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend="mjx", n_frames=n_frames)

    if STL_REWARD:
      self.reward_config = get_stl_config()
      self._H = H
    else:
      self.reward_config = get_config()
      for key, value in kwargs.items():
        if key.endswith("_scale"):
          self.reward_config.rewards.scales[key[:-6]] = value

    # The raw STL formula has no generated weighting parameters.  The argument
    # is retained for API compatibility and rejected explicitly by reward_step
    # if a caller supplies it.
    self.reward_weights = reward_weights
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
    self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
    self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
    self.uppers = jp.array([0.52, 2.1, 2.1] * 4)

    # The order below is the order consumed by the generated predicates.
    feet_site = [
        "foot_front_left",
        "foot_front_right",
        "foot_hind_left",
        "foot_hind_right",
    ]
    feet_site_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, name)
        for name in feet_site
    ]
    assert not any(idx == -1 for idx in feet_site_id), "Foot site not found."
    self._feet_site_id = np.array(feet_site_id)

    lower_leg_body = [
        "lower_leg_front_left",
        "lower_leg_front_right",
        "lower_leg_hind_left",
        "lower_leg_hind_right",
    ]
    lower_leg_body_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, name)
        for name in lower_leg_body
    ]
    assert not any(idx == -1 for idx in lower_leg_body_id), "Leg body not found."
    self._lower_leg_body_id = np.array(lower_leg_body_id)

    # Static mapping used to reduce MJX contact constraints to one normal-force
    # signal per foot.  Lower-leg geoms include the corresponding foot geom in
    # the supplied Barkour model.
    geom_bodyid = np.asarray(sys.mj_model.geom_bodyid)
    self._foot_geom_membership = jp.asarray(
        np.stack([geom_bodyid == body_id for body_id in lower_leg_body_id]),
        dtype=bool,
    )

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso"
    )
    self._foot_radius = 0.0175
    self._nv = sys.nv
    self._command_sampling = command_sampling
    self._bound_straight_commands = bound_straight_commands

  def _update_mode_hysteresis(
      self, mode: jax.Array, command: jax.Array
  ) -> jax.Array:
    vx = jp.abs(command[0])
    mode = jp.where(
        (mode == MODE_WALK) & (vx >= WALK_TO_TROT_ENTER),
        jp.array(MODE_TROT, dtype=jp.int32),
        mode,
    )
    mode = jp.where(
        (mode == MODE_TROT) & (vx <= TROT_TO_WALK_EXIT),
        jp.array(MODE_WALK, dtype=jp.int32),
        mode,
    )
    mode = jp.where(
        (mode == MODE_TROT) & (vx >= TROT_TO_BOUND_ENTER),
        jp.array(MODE_BOUND, dtype=jp.int32),
        mode,
    )
    mode = jp.where(
        (mode == MODE_BOUND) & (vx <= BOUND_TO_TROT_EXIT),
        jp.array(MODE_TROT, dtype=jp.int32),
        mode,
    )
    return mode

  def _mode_from_command(self, command: jax.Array) -> jax.Array:
    """Returns a fresh gait mode assignment without hysteresis memory."""
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
    """Samples commands with the original walk/trot/bound ranges."""
    rng, key_regime, key_vx, key_vy, key_yaw = jax.random.split(rng, 5)
    del rng
    probs = jp.array(regime_sample_probs, dtype=jp.float32)
    probs = probs / jp.sum(probs)

    if self._command_sampling == "bound_only":
      sampled_regime = jp.array(MODE_BOUND, dtype=jp.int32)
    elif self._command_sampling == "walk_trot_only":
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
    sampled_vy = _sample_uniform(key_vy, cmd_vy_range[0], cmd_vy_range[1])
    sampled_yaw = _sample_uniform(
        key_yaw, cmd_yaw_range[0], cmd_yaw_range[1]
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

  def _empty_signal_history(self) -> Dict[str, jax.Array]:
    return {key: jp.zeros((self._H,), dtype=jp.float32) for key in SIGNAL_KEYS}

  def reset(self, rng: jax.Array) -> State:
    rng, key = jax.random.split(rng)
    pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
    command = self.sample_command(key)
    mode0 = self._mode_from_command(command)
    state_info = {
        "rng": rng,
        "last_act": jp.zeros(12),
        "last_vel": jp.zeros(12),
        "command": command,
        "last_contact": jp.zeros(4, dtype=bool),
        "feet_air_time": jp.zeros(4),
        "rewards": {
            key: jp.asarray(0.0)
            for key in self.reward_config.rewards.scales.keys()
        },
        "kick": jp.array([0.0, 0.0]),
        "step": jp.array(0, dtype=jp.int32),
        "history_len": jp.array(0, dtype=jp.int32),
        "mode": mode0,
        "signal_history": self._empty_signal_history(),
    }
    obs_history = jp.zeros(15 * 34)
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    metrics = {"total_dist": jp.asarray(0.0)}
    metrics.update(state_info["rewards"])
    return State(
        pipeline_state,
        obs,
        jp.asarray(0.0),
        jp.asarray(0.0),
        metrics,
        state_info,
    )

  def _foot_velocities(self, pipeline_state: base.State) -> jax.Array:
    """Returns Cartesian velocity at each foot site in FL/FR/RL/RR order."""
    pos = pipeline_state.site_xpos[self._feet_site_id]
    feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1
    return offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

  def _foot_normal_forces(self, pipeline_state: base.State) -> jax.Array:
    """Reduces active MJX contact constraints to per-foot normal force.

    MuJoCo stores the normal component first at each contact's ``efc_address``.
    Invalid/padded contacts have a negative address and contribute zero.
    """
    contact = pipeline_state.contact
    geom1 = contact.geom1
    geom2 = contact.geom2
    valid_geom1 = geom1 >= 0
    valid_geom2 = geom2 >= 0
    safe_geom1 = jp.clip(geom1, 0, self._foot_geom_membership.shape[1] - 1)
    safe_geom2 = jp.clip(geom2, 0, self._foot_geom_membership.shape[1] - 1)
    touches_foot = (
        self._foot_geom_membership[:, safe_geom1] & valid_geom1[None, :]
    ) | (
        self._foot_geom_membership[:, safe_geom2] & valid_geom2[None, :]
    )

    address = contact.efc_address
    has_constraint = address >= 0
    safe_address = jp.clip(address, 0, pipeline_state.efc_force.shape[0] - 1)
    normal_force = jp.where(
        has_constraint, jp.maximum(pipeline_state.efc_force[safe_address], 0.0), 0.0
    )
    return jp.sum(jp.where(touches_foot, normal_force[None, :], 0.0), axis=1)

  @staticmethod
  def _roll_pitch_error(quaternion: jax.Array) -> jax.Array:
    """Returns max absolute roll/pitch in radians for a wxyz quaternion."""
    w, x, y, z = quaternion
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = jp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = jp.arcsin(sin_pitch)
    return jp.maximum(jp.abs(roll), jp.abs(pitch))

  def _current_stl_signals(
      self,
      pipeline_state: base.State,
      action: jax.Array,
      previous_action: jax.Array,
      command: jax.Array,
      contact_for_slip: jax.Array,
  ) -> Dict[str, jax.Array]:
    del action, previous_action
    torso = self._torso_idx - 1
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[torso])
    local_velocity = math.rotate(
        pipeline_state.xd.vel[torso], inv_torso_rot
    )
    local_angular_velocity = math.rotate(
        pipeline_state.xd.ang[torso], inv_torso_rot
    )
    foot_velocity = self._foot_velocities(pipeline_state)
    planar_foot_speed = jp.linalg.norm(foot_velocity[:, :2], axis=1)
    slip_velocity = jp.max(
        jp.where(contact_for_slip, planar_foot_speed, 0.0)
    )
    return {
        "velocity_error_x": local_velocity[0] - command[0],
        "velocity_error_y": local_velocity[1] - command[1],
        "velocity_error_yaw": local_angular_velocity[2] - command[2],
        "base_height": pipeline_state.x.pos[torso, 2],
        "orientation_error": self._roll_pitch_error(
            pipeline_state.x.rot[torso]
        ),
        "slip_velocity": slip_velocity,
    }

  def step(self, state: State, action: jax.Array) -> State:
    rng, cmd_rng, kick_noise = jax.random.split(state.info["rng"], 3)
    push_interval = 10
    kick_theta = jax.random.uniform(kick_noise, maxval=2 * jp.pi)
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    kick *= jp.mod(state.info["step"], push_interval) == 0
    qvel = state.pipeline_state.qvel
    qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
    state = state.tree_replace({"pipeline_state.qvel": qvel})

    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    foot_pos = pipeline_state.site_xpos[self._feet_site_id]
    foot_contact_z = foot_pos[:, 2] - self._foot_radius
    contact = foot_contact_z < 1e-3
    contact_filt_mm = contact | state.info["last_contact"]
    contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
    feet_air_time = state.info["feet_air_time"] + self.dt

    up = jp.array([0.0, 0.0, 1.0])
    torso = self._torso_idx - 1
    done = jp.dot(math.rotate(up, x.rot[torso]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[torso, 2] < 0.18

    mode = self._update_mode_hysteresis(
        state.info["mode"], state.info["command"]
    )
    current_signals = self._current_stl_signals(
        pipeline_state,
        action,
        state.info["last_act"],
        state.info["command"],
        contact_filt_cm,
    )
    signal_history = {
        key: jp.roll(state.info["signal_history"][key], -1).at[-1].set(value)
        for key, value in current_signals.items()
    }
    history_len = jp.minimum(state.info["history_len"] + 1, self._H)

    if STL_REWARD:
      reward_raw, rewards = reward_step(
          reward_input=signal_history,
          commands=state.info["command"],
          mode=mode,
          valid_len=history_len,
          weights_override=self.reward_weights,
      )
      reward = jp.clip(reward_raw * self.dt, -100.0, 1000.0)
    else:
      rewards = {
          "tracking_lin_vel": self._reward_tracking_lin_vel(
              state.info["command"], x, xd
          ),
          "tracking_ang_vel": self._reward_tracking_ang_vel(
              state.info["command"], x, xd
          ),
          "lin_vel_z": self._reward_lin_vel_z(xd),
          "ang_vel_xy": self._reward_ang_vel_xy(xd),
          "orientation": self._reward_orientation(x),
          "torques": self._reward_torques(pipeline_state.qfrc_actuator),
          "action_rate": self._reward_action_rate(
              action, state.info["last_act"]
          ),
          "stand_still": self._reward_stand_still(
              state.info["command"], joint_angles
          ),
          "feet_air_time": self._reward_feet_air_time(
              feet_air_time, first_contact, state.info["command"]
          ),
          "foot_slip": self._reward_foot_slip(
              pipeline_state, contact_filt_cm
          ),
          "termination": self._reward_termination(
              done, state.info["step"]
          ),
      }
      rewards = {
          key: value * self.reward_config.rewards.scales[key]
          for key, value in rewards.items()
      }
      reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["kick"] = kick
    state.info["last_act"] = action
    state.info["last_vel"] = joint_vel
    state.info["feet_air_time"] = feet_air_time * ~contact_filt_mm
    state.info["last_contact"] = contact
    state.info["rewards"] = rewards
    state.info["step"] += 1
    state.info["rng"] = rng
    state.info["signal_history"] = signal_history
    state.info["history_len"] = history_len
    state.info["mode"] = mode

    done_bool = done
    command_reset = done_bool | (state.info["step"] > 500)
    sampled_command = self.sample_command(cmd_rng)
    next_command = jp.where(
        command_reset, sampled_command, state.info["command"]
    )
    next_mode = jp.where(
        command_reset,
        self._mode_from_command(next_command),
        state.info["mode"],
    )
    state.info["command"] = next_command
    state.info["step"] = jp.where(command_reset, 0, state.info["step"])

    def _maybe_clear(buffer):
      return jp.where(command_reset, jp.zeros_like(buffer), buffer)

    state.info["signal_history"] = {
        key: _maybe_clear(value)
        for key, value in state.info["signal_history"].items()
    }
    state.info["history_len"] = jp.where(
        command_reset,
        jp.array(0, dtype=jp.int32),
        state.info["history_len"],
    )
    state.info["mode"] = next_mode
    state.info["last_act"] = _maybe_clear(state.info["last_act"])
    state.info["last_vel"] = _maybe_clear(state.info["last_vel"])
    state.info["last_contact"] = _maybe_clear(state.info["last_contact"])
    state.info["feet_air_time"] = _maybe_clear(state.info["feet_air_time"])
    state.info["kick"] = _maybe_clear(state.info["kick"])

    obs = self._get_obs(pipeline_state, state.info, state.obs)
    obs = jp.where(
        command_reset,
        self._get_obs(pipeline_state, state.info, jp.zeros_like(state.obs)),
        obs,
    )
    state.metrics["total_dist"] = math.normalize(x.pos[torso])[1]
    state.metrics.update(state.info["rewards"])
    return state.replace(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=jp.float32(done_bool),
    )

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    torso = self._torso_idx - 1
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[torso])
    local_rpyrate = math.rotate(
        pipeline_state.xd.ang[torso], inv_torso_rot
    )
    mode_one_hot = jax.nn.one_hot(
        state_info["mode"], 3, dtype=jp.float32
    )
    obs = jp.concatenate(
        [
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),
            state_info["command"] * jp.array([2.0, 2.0, 0.25]),
            pipeline_state.q[7:] - self._default_pose,
            state_info["last_act"],
            mode_one_hot,
        ]
    )
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info["rng"], obs.shape, minval=-1, maxval=1
    )
    return jp.roll(obs_history, obs.shape[0]).at[:obs.shape[0]].set(obs)

  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    return jp.square(xd.vel[0, 2])

  def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
    return jp.sum(jp.square(xd.ang[0, :2]))

  def _reward_orientation(self, x: Transform) -> jax.Array:
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-error / self.reward_config.rewards.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    reward = jp.sum((air_time - 0.1) * first_contact)
    return reward * (math.normalize(commands[:2])[1] > 0.05)

  def _reward_stand_still(
      self, commands: jax.Array, joint_angles: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _reward_foot_slip(
      self, pipeline_state: base.State, contact_filt: jax.Array
  ) -> jax.Array:
    foot_vel = self._foot_velocities(pipeline_state)
    return jp.sum(
        jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))
    )

  def _reward_termination(
      self, done: jax.Array, step: jax.Array
  ) -> jax.Array:
    return done & (step < 500)

  def render(
      self,
      trajectory: List[base.State],
      camera: str | None = None,
      width: int = 240,
      height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or "track"
    return super().render(
        trajectory, camera=camera, width=width, height=height
    )


# envs.register_environment("barkour", BarkourEnv)
# %%
