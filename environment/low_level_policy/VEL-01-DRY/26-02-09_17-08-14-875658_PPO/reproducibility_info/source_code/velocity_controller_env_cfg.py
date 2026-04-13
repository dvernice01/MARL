# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


@configclass
class VelocityControllerEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 4
    observation_space = 13
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # robot(s)
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # custom parameters/scales
    # - action scale
    thrust_to_weight = 1.9
    moment_scale = 0.01
    
    # - command ranges
    lin_vel_ref_range = [-1.0, 1.0] # [m/s]
    yaw_vel_ref_range = [-1.0, 1.0] # [rad/s]
    canonical_interval = 30 # episodes
    resample_command_interval_s = 3.0 # [s] Average time between command changes

    # - reward scales
    rew_scale_alive = 0.1
    rew_scale_lin_vel_tracking = -30.0
    rew_scale_ang_vel_tracking = -15.5
    rew_scale_action_reg = 0.0
    rew_scale_tilt_penalty = -2000.0
    huber_delta = 0.5
    
    # - reset conditions
    max_tilt_rad = 1.4 # approx 80 degrees
    min_z_pos = 0.1 # [m]
    initial_z_min = 1.5 # [m]
    initial_z_max = 3 # [m]