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

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=15, env_spacing=2.5, replicate_physics=True
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
    safety_radius: float = 0.2                          
   
    # ── Exploration (PDF-style v_t = γ·exp(-δ·N_t)) ──────────────────────
    exploration_gamma: float = 0.1
    exploration_delta: float = 0.01

    # ── Collision termination ─────────────────────────────────────────────
    collision_distance: float = 0.25  
  
    # Drone-mounted 360x90 LiDAR raycaster — feeds the OCC map.

    ray_caster: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period= 1.0 * decimation / 120,  # match LiDAR update rate (10 Hz)
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        mesh_prim_paths=[],  # populated at runtime
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-45.0, 45.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),
        debug_vis=False,
    )


    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",  # aggiungere o togliere /Robot/body prima di front_cam in base a se stai collezionando dati o facendo RL
        update_period=1.0 * decimation / 120,
        update_latest_camera_pose=True,  
        height=67,   # keep small for RL — 480x640 will crush perf at 4096 envs
        width=120,
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
    lin_vel_reward_scale = 0.1
    ang_vel_reward_scale = 0.1
    rew_scale_action_reg = 0.1
    alive_reward_scale = 0.1
    death_reward_scale = -50.0
    distance_to_obstacles_reward_scale = 0.0
    exploration_reward_scale = 0.1


    # ── Goal-distance reward (Kulkarni & Alexis 2024, eq. 2) ─────────────
    # r1, r2: two Gaussian kernels exp(-d^2/nu) at different widths
    goal_nu1: float = 4.0            # narrow: fine final approach (~1-2 m)
    goal_nu2: float = 16.0          # broad: long-range pull (meaningful at 10-20 m)
    goal_lambda1: float = 15.0
    goal_lambda2: float = 10.0
    # r3: linear normalized proximity 1 - d/nu3 (nu3 >= map diagonal ~47 m)
    # goal_nu3: float = 47.0
    # goal_lambda3: float = 10.0
    # r4: progress / delta-distance reward, weight on (d_{t-1} - d_t)
    goal_progress_scale: float = 30.0

    # ── Goal reaching / success ──────────────────────────────────────────
    goal_radius: float = 1.0
    exploration_stop: float = 2.0

    # ── Curriculum: expanding spawn box for drone and goal ───────────────
    curriculum_window: int = 100             # completed episodes per evaluation
    curriculum_success_threshold: float = 0.70
    min_goal_separation: float = 1.5         # min drone-goal distance at spawn (> goal_radius)
    curriculum_max_resample: int = 10        # resample tries to satisfy separation
    spawn_z_min: float = 1.0           # aisle altitude band
    spawn_z_max: float = 8.0


    spawn_clearance: float = 0.5            # required clearance (m) from obstacles at spawn (> collision_distance)
    occupancy_map_path: str = "/workspace/environment/uav_navigation/source/uav_navigation/uav_navigation/tasks/direct/uav_navigation/occupancy_3d.npy"
    occupancy_meta_path: str = "/workspace/environment/uav_navigation/source/uav_navigation/uav_navigation/tasks/direct/uav_navigation/occupancy_3d_meta.npy"

    # Evaluation override: if set to (x, y), force the drone spawn to this world
    # position (z = middle of the spawn band) instead of curriculum sampling.
    # Leave None for normal training.
    eval_fixed_spawn_xy: tuple | None = None

    goal_success_thresholds: tuple = (5.0, 4.0, 3.0, 2.0, 1.0)
    goal_success_scale: float = 5.0     # one-time per crossing; not multiplied by step_dt
    curriculum_near_radius: float = 2.5
    curriculum_near_levels: int = 2


