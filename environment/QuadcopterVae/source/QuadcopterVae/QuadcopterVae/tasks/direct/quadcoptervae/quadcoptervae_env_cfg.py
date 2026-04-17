# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
# Imports for Camera
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

class QuadcoptervaeEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcoptervaeEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcoptervaeEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 50.0
    decimation= 10
    decimation_low_level = 2
    action_space = 4
    observation_space = 11
    state_space = 0
    debug_vis = True

    #low_level_network_path: str = "/workspace/environment/low_level_policy/VEL-01-DRY/26-02-09_17-08-14-875658_PPO/checkpoints/best_agent.pt"
    max_velocity: float = 1.0
    max_yaw_rate: float = 1.0
    
    ui_window_class_type = QuadcoptervaeEnvWindow

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

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-4.0, -2.0, 5.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # sensors
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",  # 'body' is the base link of Crazyflie
        update_period=0.1,
        height=270,   # keep small for RL — 480x640 will crush perf at 4096 envs
        width=480,
        data_types=["distance_to_camera"],  # depth sensing
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.04, # per essere coerente con VAE
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0), # lim_max = 10 m
        ),
        offset=CameraCfg.OffsetCfg(
            #pos=(0.05, 0.0, 0.0),   # slightly in front of the drone body
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),  # forward-facing, ROS convention
            convention="ros",
        ),
    )

    # reward scales
    lin_vel_reward_scale = 0.05
    ang_vel_reward_scale = 0.01
    distance_to_goal_reward_scale = 15.0
    alive_reward_scale = 0.1
    death_reward_scale = -5.0