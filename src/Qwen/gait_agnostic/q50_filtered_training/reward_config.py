"""Reward configuration for the q50-retained gait-agnostic environment."""

from ml_collections import config_dict


def get_config():
  rewards = config_dict.ConfigDict(
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
  return config_dict.ConfigDict(dict(rewards=rewards))


def get_stl_config():
  """Metric schema for the q50-filtered retained-component reward."""
  scales = config_dict.ConfigDict(
      dict(
          total_stl_reward=1.0,
          rho_task_safety=1.0,
          vel_track_x=1.0,
          safe_height=1.0,
          safe_orientation=1.0,
          diagnostic_abs_velocity_error_y=0.0,
          diagnostic_abs_velocity_error_yaw=0.0,
      )
  )
  return config_dict.ConfigDict(
      dict(rewards=config_dict.ConfigDict(dict(scales=scales)))
  )
