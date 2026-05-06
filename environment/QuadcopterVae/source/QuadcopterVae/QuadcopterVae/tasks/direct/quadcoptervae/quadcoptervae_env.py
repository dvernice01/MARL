# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym

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
import isaaclab.utils.math as math_utils

from .quadcoptervae_env_cfg import QuadcoptervaeEnvCfg
from isaaclab_assets import CRAZYFLIE_CFG  
from isaaclab.markers import CUBOID_MARKER_CFG  
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import transform_points, unproject_depth, quat_inv, quat_apply
import matplotlib.pyplot as plt
import numpy as np
import os
from .vae_residual_batch import VAE
import inspect
print("VAE loaded from:", inspect.getfile(VAE))
from tqdm import tqdm
import cv2

import wandb

wandb.login()

# Project that the run is recorded to
project = "quadcoptervae"

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(13, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 4),
        )
    def forward(self, x):
        return self.net(x)

class vae_config:
    use_vae = True
    latent_dims = 512
    #032824
    model_file = (
        "/workspace/vae_container/Vae/checkpoint/vae_best_20260505_055654.pt"
    )
    model_folder = "/workspace/vae_container/Vae/checkpoint"
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = False

class CollisionImage:
    def __init__(self):
        self.CX = 240.0
        self.CY = 135.0
        self.FX = 252.91646
        self.FY = 252.91646
        self.MAX_DEPTH = 10.0
        self.MIN_DEPTH = 0.2
        self.ROBOT_EDGE_LEN = 0.2   # Crazyflie: cube side = 2r, r = 0.1m
        self.OFFSET_DIST    = 0.1
        self.H = 270
        self.W = 480
        self.MESHGRID = self.create_meshgrid(self.H, self.W, self.CX, self.CY, self.FX, self.FY)
        
        
    def sanitize_depth(self, depth):
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0
        depth[depth > self.MAX_DEPTH] = self.MAX_DEPTH
        return depth

    def create_meshgrid(self, H, W, cx, cy, fx, fy):
        x = np.arange(0, H, dtype=np.float32)
        y = np.arange(0, W, dtype=np.float32)
        x, y = np.meshgrid(y, x)
        z = np.ones((H, W))
        x = (x - cx) / fx
        y = (y - cy) / fy
        return np.stack([x, y, z], axis=0)
    
    def process_image_like_author(self, image):
        image = image.copy()
        image[image < self.MIN_DEPTH] = -1.0
        image[image > self.MAX_DEPTH] = self.MAX_DEPTH
        image = image * 0.1          # scale to [0, 1]
        image[image < 0.0] = 0.0
        image[image > 1.0] = 1.0
        return image

    def detect_edges(self, depth_uint8, depth_float):
        edge_image = cv2.Canny(depth_uint8, 30, 50)
        edges_rc   = np.where(edge_image > 0)
        edges      = np.array(list(zip(edges_rc[0], edges_rc[1])))
        if len(edges) == 0:
            return edges, edge_image
        for i in range(len(edges)):
            edge = edges[i]
            neighbors = [
                (max(edge[0]-1,0), edge[1]), (min(edge[0]+1,self.H-1), edge[1]),
                (edge[0], max(0,edge[1]-1)), (edge[0], min(self.W-1,edge[1]+1)),
                (max(edge[0]-2,0), edge[1]), (min(edge[0]+2,self.H-1), edge[1]),
                (edge[0], max(0,edge[1]-2)), (edge[0], min(self.W-1,edge[1]+2)),
            ]
            min_d = depth_float[edge[0], edge[1]]
            if min_d <= 0.0:
                for j in neighbors:
                    if depth_float[j[0], j[1]] > 0.0:
                        min_d = depth_float[j[0], j[1]]
                        edge  = j
                        break
            for j in neighbors:
                if 0.1 < depth_float[j[0], j[1]] < min_d:
                    min_d = depth_float[j[0], j[1]]
                    edges[i] = j
            if min_d < depth_float[edge[0], edge[1]] and depth_float[edge[0], edge[1]] > 0.1:
                edges[i] = (0,0)
        return edges, edge_image

    def build_D_M_from_cubes(self, edges, point_cloud, depth_float,
                            edge_length):
        """
        Replicates the author's create_cube_mesh + Warp ray casting without Warp.

        The author places a cube of side `edge_length` at each edge point in 3D,
        then renders a depth image of all cubes from the camera.

        We replicate this by:
        1. Taking every 5th edge pixel (same as author: edges[::5])
        2. Getting its 3D position from the point_cloud
        3. Computing how large that cube appears in the image
            (cube side in pixels = edge_length * fx / Z)
        4. Painting a square of that size in D_M with depth value Z
            (this is what the ray caster would return for rays hitting the cube)

        A cube projects as a square on the image plane (not a circle),
        so we fill a square patch — no mgrid/circle needed.
        """
        D_M = np.full((self.H, self.W), self.MAX_DEPTH, dtype=np.float32)

        # sample every 5th edge — exact replication of author's edges[::5]
        sampled_edges = edges[::5]

        for i in range(len(sampled_edges)):
            # pixel coordinates of this edge
            v = int(sampled_edges[i, 0])   # row
            u = int(sampled_edges[i, 1])   # col

            # 3D position of this edge pixel from the point cloud
            # point_cloud shape: (3, H, W) → [x, y, z] at pixel (v, u)
            X = point_cloud[0, v, u]
            Y = point_cloud[1, v, u]
            Z = point_cloud[2, v, u]   # this is the depth z of the edge point

            # skip invalid points
            if Z < self.MIN_DEPTH or not np.isfinite(Z):
                continue

            # the cube has side edge_length in meters
            # its half-side projected onto the image plane at depth Z:
            #   half_side_px = (edge_length / 2) * fx / Z
            half_px_u = int((edge_length / 2.0) * self.FX / Z)
            half_px_v = int((edge_length / 2.0) * self.FY / Z)

            # clamp to reasonable size
            half_px_u = max(1, min(half_px_u, 60))
            half_px_v = max(1, min(half_px_v, 60))

            # image bounds of the projected cube face
            v0 = max(0, v - half_px_v)
            v1 = min(self.H, v + half_px_v + 1)
            u0 = max(0, u - half_px_u)
            u1 = min(self.W, u + half_px_u + 1)

            if v1 <= v0 or u1 <= u0:
                continue

            # paint the square patch with depth Z
            # np.minimum keeps the closest cube if two overlap
            D_M[v0:v1, u0:u1] = np.minimum(D_M[v0:v1, u0:u1], Z)

        return D_M

    def depth_to_collision_image(self,depth_raw: np.ndarray) -> np.ndarray:
        depth = self.sanitize_depth(depth_raw)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape != (self.H, self.W):
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            depth = self.sanitize_depth(depth)

        # D_offset — equation 5
        x = self.MESHGRID[0] * depth
        y = self.MESHGRID[1] * depth
        z = self.MESHGRID[2] * depth
        range_img   = np.sqrt(x**2 + y**2 + z**2)
        range_img   = np.nan_to_num(range_img, nan=self.MAX_DEPTH)
        safe_inv = np.divide(self.OFFSET_DIST, range_img, 
                     out=np.zeros_like(range_img), 
                     where=range_img > 0)
        z_offset = np.where(range_img > 0, (1 - safe_inv) * z, 0.0)
        norm_offset = self.process_image_like_author(z_offset)

        # edge detection
        depth_uint8 = (depth / self.MAX_DEPTH * 255).astype(np.uint8)
        edges, _    = self.detect_edges(depth_uint8, depth)
        if len(edges) < 10:
            return norm_offset

        # build point cloud for cube placement
        point_cloud = np.stack([x, y, z], axis=0)   # (3, H, W)

        # D_M — cube mesh approximation
        D_M_raw    = self.build_D_M_from_cubes(edges, point_cloud, depth,
                                        edge_length=self.ROBOT_EDGE_LEN)
        norm_D_M   = self.process_image_like_author(D_M_raw)

        # equation 6
        collision  = np.minimum(norm_offset, norm_D_M)
        collision  = np.nan_to_num(collision, nan=0.0)
        return collision

    def clean_state_dict(self, state_dict):
        clean_dict = {}
        for key, value in state_dict.items():
            if "module." in key:
                key = key.replace("module.", "")
            if "dronet." in key:
                key = key.replace("dronet.", "encoder.")
            clean_dict[key] = value
        return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device
        self.collision = CollisionImage()
        #self.collision.__init__()
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims,inference_mode = True).to(self.device)
        # combine module path with model file name
        weight_file_path = self.config.model_file
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = self.collision.clean_state_dict(torch.load(weight_file_path))
        for k, v in state_dict.items():
            print(f"{k}: {v.shape}")
        missing, unexpected = self.vae_model.load_state_dict(state_dict, strict=False)
        #print(state_dict.keys())
        print("Missing keys:   ", missing)
        print("Unexpected keys:", unexpected)
        core_keys = [k for k in missing if "conv" in k or "dense0" in k]
        if core_keys:
            raise RuntimeError(f"Core architecture mismatch: {core_keys}")
        self.vae_model.eval()
        self.max_depth = 10.0

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            #image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            if image_tensors.ndim == 4:  # (N, 1, H, W)
                pass
            elif image_tensors.ndim == 3:  # (N, H, W)
                image_tensors = image_tensors.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected shape: {image_tensors.shape}")

            # c'è gia nel preprocess ma lo tengo per maggiore controllo        
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            #interpolated_image = interpolated_image.float()
            z_sampled, means, log_var = self.vae_model.encode(interpolated_image)
            n_clipped = ((log_var < -10) | (log_var > 4)).sum().item()
            if n_clipped > 0:
                print(f"WARNING: {n_clipped} log_var values were clamped")
            # print("means  min/max:", means.min().item(), means.max().item())
            # print("z_samp min/max:", z_sampled.min().item(), z_sampled.max().item())
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims
    
    def _preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, 0, self.max_depth)
        depth = depth / self.max_depth
        return depth
    
    def preprocess(self, depth_torch: torch.Tensor) -> torch.Tensor:
        """
        depth_torch: (N, H, W, 1)
        returns: (N, 1, H, W)
        """

        depth_np = depth_torch.squeeze(-1).cpu().numpy()  # (N,H,W)

        collision_batch = []
        for i in range(depth_np.shape[0]):
            collision = self.collision.depth_to_collision_image(depth_np[i])
            collision_batch.append(collision)

        collision_np = np.stack(collision_batch, axis=0)  # (N,H,W)

        collision = torch.from_numpy(collision_np).float().to(self.device)

        collision = collision.unsqueeze(1)  # (N,1,H,W)

        # resize if needed
        if collision.shape[-2:] != self.config.image_res:
            collision = torch.nn.functional.interpolate(
                collision,
                size=self.config.image_res,
                mode=self.config.interpolation_mode,
            )

        return collision

    def _get_depth_data_in_body_frame(self, max_depth) -> dict:
        """
        Returns a dict with the raw depth tensor and optional pointcloud.

        Depth tensor shape : (num_envs, H, W, 1)  dtype: torch.float32
        Values represent distance in METERS along the camera Z-axis.
        Invalid / out-of-range pixels are filled with torch.inf or 0.0
        depending on the clipping_range setting.
        """
        # ------------------------------------------------------------------
        # RAW DEPTH IMAGE
        # shape: (num_envs, H, W, 1), float32, unit: meters
        # "distance_to_image_plane" = z-axis depth (standard pinhole model)
        # "distance_to_camera"      = euclidean ray distance
        # ------------------------------------------------------------------

        depth: torch.Tensor = self.camera.data.output["distance_to_camera"]

        # ------------------------------------------------------------------
        # CLEAN invalid values (inf / nan → max_range, useful before feeding to NN)
        # ------------------------------------------------------------------
        max_range = self.cfg.camera.spawn.clipping_range[1]
        depth= torch.nan_to_num(depth, nan=max_range, posinf=max_range, neginf=max_range)
        depth = torch.clamp(depth, 0, max_depth)
        depth_clean = depth / max_depth

        # ------------------------------------------------------------------
        # 2. CAMERA INTRINSICS & POSE (world frame)
        #    intrinsic_matrices : (num_envs, 3, 3)
        #    pos_w              : (num_envs, 3)   camera origin in world
        #    quat_w_world       : (num_envs, 4)   camera orientation in world
        #                         quaternion convention: (w, x, y, z)
        # ------------------------------------------------------------------
        K   = self.camera.data.intrinsic_matrices   # (N, 3, 3)
        cam_pos_w  = self.camera.data.pos_w         # (N, 3)
        cam_quat_w = self.camera.data.quat_w_world  # (N, 4)  w-first


        # shape: (num_envs, H*W, 3) in camera frame
        # points_cam = unproject_depth(
        #     depth_clean,                          # (N, H, W, 1)
        #     self.camera.data.intrinsic_matrices,  # (N, 3, 3)
        # )
        # transform to world frame using camera pose
        points_world = transform_points(
            depth_clean,
            self.camera.data.pos_w,   # (N, 3) camera position in world
            self.camera.data.quat_w_world,  # (N, 4) camera orientation
        )  # shape: (N, H*W, 3)

        body_pos_w  = self._robot.data.root_pos_w    # (N, 3)
        body_quat_w = self._robot.data.root_quat_w   # (N, 4)

        # Translate points so the body origin is at zero
        points_centered = points_world - body_pos_w.unsqueeze(1)  

        # Rotate from world frame into body frame using inverse body quaternion
        body_quat_inv = quat_inv(body_quat_w)                     # (N, 4)

        # quat_apply broadcasts over the point dimension
        # expand quat to match (N, H*W, 4) for batched rotation
        body_quat_inv_exp = body_quat_inv.unsqueeze(1).expand(-1, points_centered.shape[1], -1)
        points_body = quat_apply(body_quat_inv_exp, points_centered) 

        # ------------------------------------------------------------------
        # 7. CAMERA ORIGIN in BODY FRAME
        #    Same transform applied to the single camera position vector
        # ------------------------------------------------------------------
        cam_centered = (cam_pos_w - body_pos_w)                   # (N, 3)
        cam_pos_body = quat_apply(body_quat_inv, cam_centered)    # (N, 3)

        return points_body, cam_pos_body

class QuadcoptervaeEnv(DirectRLEnv):
    cfg: QuadcoptervaeEnvCfg

    def __init__(self, cfg: QuadcoptervaeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.low_level_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        #self.decimator_low_level = 2
        self.target_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_yaw_cmd = torch.zeros(self.num_envs, 1, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.rel_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Vae init
        self.vae_encoder = VAEImageEncoder(vae_config, device=self.device)
        self.max_depth = 10.0  
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "life",
                "died",
                "time_out",
                "final_distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # AGGIUNGO DEGLI INIT CHE MI SERVIRANNO
        self.distance_to_bounds_x = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_bounds_y = torch.zeros(self.num_envs, device=self.device)
        self.final_distance_to_goal_b = torch.zeros(self.num_envs, device=self.device)
        # INIT PER CURRICULUM LEARNING
        self.curriculum_level = 0
        self.arena_size = torch.ones(self.num_envs, device=self.device) * 30

        # Accumulatori per valutazione
        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0.0

        # Reward massimo teorico per episodio (per normalizzare)
        max_theoretical_reward = self.cfg.alive_reward_scale * self.cfg.distance_to_goal_reward_scale * self.max_episode_length_s
        self.policy_network = self._load_policy_network()

    def _load_policy_network(self):
        checkpoint = torch.load(
            "/workspace/environment/quadcopter_vel_control/projects/quadcopter_vel_control/logs/skrl/quadcopter_vel_control_direct/2026-04-08_08-45-39_Manual_PPO/26-04-08_08-45-43-335682_PPO/checkpoints/best_agent.pt",
            map_location=self.device
        )

        preprocessor = checkpoint["state_preprocessor"]
        self.ll_running_mean     = preprocessor["running_mean"].float().to(self.device)
        self.ll_running_variance = preprocessor["running_variance"].float().to(self.device)
        self.ll_epsilon          = 1e-8 # skrl default
        self.ll_clip_threshold   = 5.0  # skrl default

        policy = PolicyNet()
        missing, unexpected = policy.load_state_dict(checkpoint["policy"], strict=False)

        # print(f"Missing keys:    {missing}")      # should only be log_std_parameter
        # print(f"Unexpected keys: {unexpected}")   # should be empty

        assert "net.0.weight" not in missing, "Core weights failed to load!"

        policy.to(self.device)
        policy.eval()
        return policy
        
    def _normalize_ll_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # Exact formula from the docs:
        # clip((x - mean) / sqrt(variance + epsilon), -c, c)
        normalized = (obs - self.ll_running_mean) / torch.sqrt(self.ll_running_variance + self.ll_epsilon)
        return torch.clamp(normalized, -self.ll_clip_threshold, self.ll_clip_threshold)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # Camera sensor setup
        self.camera = Camera(self.cfg.camera)
        #self.ray_caster = RaycasterSensorSceneCfg
        self.scene.sensors["camera"] = self.camera
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # self.target_vel_cmd[:,:3] = self._actions[:, :3]
        # self.target_yaw_cmd = self._actions[:, 3]
        vel_zero = torch.zeros(self.num_envs, 4, device=self.device)
        #vel_zero[:,3] = 0.3
        vel_zero[:,0] = 1.0
        self.target_vel_cmd[:,:3] = vel_zero[:,:3]
        self.target_yaw_cmd = vel_zero[:,3]


    def _apply_action(self):

        if self.common_step_counter % self.cfg.decimation_low_level == 0:

            low_level_obs = torch.hstack((
                self._robot.data.root_lin_vel_b,  # 3
                self._robot.data.root_ang_vel_b,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self.target_vel_cmd,  # 3
                self.target_yaw_cmd.reshape(-1, 1),  # 1
            ))
            low_level_obs = low_level_obs.to(self.device).to(torch.float32)
            
            # Inference with frozen policy
            with torch.no_grad():
                obs_norm = self._normalize_ll_obs(low_level_obs)
                self.low_level_actions = self.policy_network(obs_norm)

            self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.low_level_actions[:, 0] + 1.0) / 2.0
            self._moment[:, 0, :] = self.cfg.moment_scale * self.low_level_actions[:, 1:]

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        self.rel_pos_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w, 
                self._robot.data.root_quat_w, 
                self._desired_pos_w
            )
        self.final_distance_to_goal = torch.linalg.norm(self.rel_pos_b, dim=1)
        depth = self.camera.data.output["distance_to_camera"]  # (N, H, W, 1)

        # preprocess (VERY IMPORTANT)
        collision = self.vae_encoder.preprocess(depth)
        assert not torch.isnan(collision).any(), f"NaN in collision: {collision.min()}, {collision.max()}"
        assert not torch.isinf(collision).any(), f"Inf in collision: {collision.min()}, {collision.max()}"
    
        # encode
        latent = self.vae_encoder.encode(collision)  # (N, latent_dim)

        if self.common_step_counter % 20 == 0:
            depth_vis = depth[0, :, :, 0] / self.max_depth
            depth_np = depth_vis.detach().cpu().numpy()
            plt.imshow(depth_np, cmap='plasma', vmin=0, vmax=1)
            plt.colorbar(label='Depth (m)')
            plt.title('Raw Depth')
            plt.savefig('depth_check.png')
            plt.close()

            collision_np = collision[0, 0].detach().cpu().numpy()
            plt.imshow(collision_np, cmap='plasma', vmin=0, vmax=1)
            plt.title('Collision Image')
            plt.savefig('collision_check.png')
            plt.close()

            recon = self.vae_encoder.decode(latent)
            recon_np = recon[0, 0].detach().cpu().numpy()
            plt.imshow(recon_np, cmap='plasma', vmin=0, vmax=1)
            plt.title('Reconstruction')
            plt.savefig('recon_check.png')
            plt.close()

            # print("latent min/max:", latent.min().item(), latent.max().item())
            # print("recon  min/max:", recon.min().item(),  recon.max().item())
            # print("collision min/max:", collision.min().item(), collision.max().item())

        # for name, param in self.vae_encoder.vae_model.named_parameters():
        #     if "mu" in name or "log_var" in name or "fc" in name:
        #         print(name, param.shape)

        obs = torch.cat(
            [
                self.rel_pos_b,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self.distance_to_bounds_x.reshape(-1, 1),
                self.distance_to_bounds_y.reshape(-1, 1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        origins = self.scene.env_origins
        lin_vel_sum = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sum = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        lin_vel = 1 - torch.tanh(lin_vel_sum / 0.8)
        ang_vel = 1 - torch.tanh(ang_vel_sum / 0.8)
        self._episode_sums["final_distance_to_goal"] += self.final_distance_to_goal * self.step_dt
        distance_to_goal_mapped = 1 - torch.tanh(self.final_distance_to_goal / 0.8)
        
        square_side = self.arena_size  
        relative_distance_to_bounds_x = torch.abs((origins[:, 0]) - self._robot.data.root_pos_w[: ,0])
        relative_distance_to_bounds_y = torch.abs((origins[:, 1]) - self._robot.data.root_pos_w[: ,1])
        self.distance_to_bounds_x = (square_side / 2) - relative_distance_to_bounds_x
        self.distance_to_bounds_y = (square_side / 2) - relative_distance_to_bounds_y

        is_alive = torch.logical_and(self.distance_to_bounds_x >= 0.0, self.distance_to_bounds_y >= 0.0)
        life = torch.where(is_alive, 
                        self.cfg.alive_reward_scale, 
                        self.cfg.death_reward_scale)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "life": life * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = (self._robot.data.root_pos_w[:, 2] < 0.1) | \
                (self._robot.data.root_pos_w[:, 2] > 7.9) | \
                (self.distance_to_bounds_x < 0.0) | \
                (self.distance_to_bounds_y < 0.0)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        #Update Episode Sums for the reset batch
        num_deaths = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        num_timeouts = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        self._episode_sums["died"] += num_deaths   
        self._episode_sums["time_out"] += num_timeouts
        #self._episode_sums["world pos z"] += self._robot.data.root_pos_w[:, 2] * self.step_dt
        #self._episode_sums["distance to bound x"] += self.distance_to_bounds_x * self.step_dt
        #self._episode_sums["distance to bound y"] += self.distance_to_bounds_y * self.step_dt

        self.extras["log"] = dict()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # Separate actual reward components from diagnostic metrics
            if key in ["died", "time_out"]:
                extras["Episode_Termination/" + key] = episodic_sum_avg / self.max_episode_length_s
            elif key in ["final_distance_to_goal"]:
                extras["Episode_Info/" + key] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # azzerato DOPO aver letto
        self.extras["log"].update(extras)

        #RESET DEGLI ENVIRONMENTS
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-15.0, 15.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 5.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] = -4.0
        default_root_state[:, 1] = -2.0
        default_root_state[:, 2] = 2.0
        default_root_state[:, -1] = 90
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

