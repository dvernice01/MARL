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
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

class UavNavigationEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: UavNavigationEnv, window_name: str = "IsaacLab"):
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
class UavNavigationEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 100.0
    decimation= 10
    decimation_low_level = 2
    action_space = 4
    observation_space = 719 
    state_space = 0
    debug_vis = True

    max_velocity: float = 1.0
    max_yaw_rate: float = 1.0
    
    ui_window_class_type = UavNavigationEnvWindow

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
        collision_group=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=20, env_spacing=2.5, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01
    
    # ── Online occupancy mapping ─────────────────────────────────────────
    occ_cell_size: float = 0.25
    occ_samples_per_ray: int = 40   

    # ── Distance-to-obstacles (smooth safety margin reward) ──────────────
    safety_radius: float = 0.5                          
   
    # ── Exploration (PDF-style v_t = γ·exp(-δ·N_t)) ──────────────────────
    exploration_gamma: float = 1.0
    exploration_delta: float = 0.01

    # ── Collision termination ─────────────────────────────────────────────
    collision_distance: float = 0.3  
  
    # Drone-mounted 360x90 LiDAR raycaster — feeds the OCC map.

    ray_caster: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period= 1.0 * decimation / 120,  # match LiDAR update rate (10 Hz)
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        mesh_prim_paths=[],  # populated at runtime
        pattern_cfg=patterns.LidarPatternCfg(
            channels=24,
            vertical_fov_range=(-45.0, 45.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        debug_vis=False,
    )


    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",  # aggiungere o togliere /Robot/body prima di front_cam in base a se stai collezionando dati o facendo RL
        update_period=1.0 * decimation / 120,
        update_latest_camera_pose=True,  
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
    lin_vel_reward_scale = 0.0
    ang_vel_reward_scale = 0.0
    distance_to_goal_reward_scale = 50.0
    rew_scale_action_reg = 0.1
    alive_reward_scale = 0.1
    death_reward_scale = -50.0
    distance_to_obstacles_reward_scale = 5.0
    exploration_reward_scale = 5.0