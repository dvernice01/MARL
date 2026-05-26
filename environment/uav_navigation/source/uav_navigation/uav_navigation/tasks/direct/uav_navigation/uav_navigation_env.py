# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Buonds full_warehouse:
X: -28.00 → 8.00  (width=36.00m)
Y: -41.40 → 33.42  (depth=74.82m)
Z: -0.01 → 9.30  (height=9.31m)
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn.functional as F
import numpy as np
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
from isaaclab.utils.math import subtract_frame_transforms, transform_points, unproject_depth, quat_inv, quat_apply
import isaaclab.utils.math as math_utils

from .uav_navigation_env_cfg import UavNavigationEnvCfg
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.markers import CUBOID_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCaster, RayCasterCfg
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.mplot3d import Axes3D
from .vae_residual_batch import VAE
from .autoencoder_3d import AE3D
import inspect
from tqdm import tqdm
import cv2
import omni.usd
from pxr import UsdGeom

wandb.login()

project = "uav_navigation"

# ── Paths ──────────────────────────────────────────────────────────────────────
LOCAL_MAPS_SAVE_DIR = "/workspace/environment/uav_navigation/outputs/local_maps_uav_navigation"  # where to save local maps; set to None to disable
LOCAL_MAP_SAVE_EVERY = 1000   # steps between saves; set to 0 to disable
LOCAL_NZ = 8                  # local map depth  (z axis)
LOCAL_NY = 16                 # local map height (y axis)
LOCAL_NX = 16                 # local map width  (x axis)
MIN_ALIVE_STEPS_TO_SAVE = 0   # consecutive alive steps required before saving a map
LOCAL_MAP_START_STEP = 0
LOCAL_CELL_SIZE = 0.25
ONLINE_OCC_VIZ_EVERY = 500    # steps between online-OCC sanity plots; 0 to disable


class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim=13, output_dim=4, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class vae_config:
    use_vae = True
    latent_dims = 512
    #032824
    model_file = (
        "/workspace/vae_container/Vae/runs/xve5n1nh/brisk-sweep-1/checkpoints/vae_best_20260515_091157.pt"
    )
    model_folder = "/workspace/vae_container/Vae/checkpoint"
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = False

class ae3d_config:
    """
    Mirror of the architectural hyperparams used to train the 3D AE.
    Values MUST match the wandb config of run 8ivrdugy, otherwise the
    state_dict will not load.
    """
    use_ae3d = True
    latent_dim = 192                
    model_file = (
        "/workspace/vae_container/Vae/runs/8ivrdugy/daily-sweep-1/"
        "checkpoints/ae3d_best_20260525_145127.pt"
    )
    latent_dim         = 192
    num_conv_layers    = 6
    num_deconv_layers  = 5
    use_residual       = True
    residual_every     = 2
    use_skip           = True
    svs_max            = 1e-8  
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
        self.vae_model = VAE(
            input_dim          = 1,
            latent_dim         = 512,
            with_logits        = False,
            inference_mode     = False,
            num_conv_layers    = 6,
            use_residual       = True,
            num_deconv_layers  = 7,
            residual_every     = 2,
            use_skip           = True,
            #decoder_num_dense = cfg.decoder_num_dense,
        ).to(device)
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
    
class autoencoder_3d:
    """
    Wraps the trained 3D AE (OCC + SVS local maps → latent).

    Mirrors VAEImageEncoder in shape: same preprocess/encode/get_latent_dim
    methods, so wiring into _get_observations is symmetric.
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device

        self.model = AE3D(
            input_dim         = 2,
            latent_dim        = config.latent_dim,
            with_logits       = True,
            inference_mode    = True,
            num_conv_layers   = config.num_conv_layers,
            use_residual      = config.use_residual,
            residual_every    = config.residual_every,
            num_deconv_layers = config.num_deconv_layers,
            use_skip          = config.use_skip,
        ).to(device)

        print(f"[AE3DEncoder] Loading weights from {config.model_file}")
        state_dict = torch.load(config.model_file, map_location=device)
        # Strip a possible "module." prefix from DataParallel-trained checkpoints.
        clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        print(f"[AE3DEncoder] Missing keys:    {missing}")
        print(f"[AE3DEncoder] Unexpected keys: {unexpected}")
        core = [k for k in missing if "conv" in k or "dense" in k]
        if core:
            raise RuntimeError(f"3D AE state_dict mismatch on core layers: {core}")
        self.model.eval()

        self.env = env
        device = env.device

        self.occ_cell_size = float(env.cfg.occ_cell_size)
        self.occ_origin = torch.tensor(
            [env.x_min, env.y_min, env.z_min],
            dtype=torch.float32, device=device,
        )
        self.NX = int(math.ceil((env.x_max - env.x_min) / self.occ_cell_size))
        self.NY = int(math.ceil((env.y_max - env.y_min) / self.occ_cell_size))
        self.NZ = int(math.ceil((env.z_max - env.z_min) / self.occ_cell_size))
        self.occ_map_dims = (self.NZ, self.NY, self.NX)

        # ── Persistent per-env buffers, populated ONLINE from sensors ──────
        # global_occ_map: any cell that has ever received a LiDAR hit is 1.
        # global_visit_counts: cell counter incremented at the drone's pose.
        # Both wiped on _reset_idx.
        self.global_occ_map = torch.zeros(
            (env.num_envs, self.NZ, self.NY, self.NX),
            dtype=torch.uint8, device=device,
        )
        self.global_visit_counts = torch.zeros(
            (env.num_envs, self.NZ, self.NY, self.NX),
            dtype=torch.float32, device=device,
        )

        # ── Local map shape ────────────────────────────────────────────────
        self.local_nz = LOCAL_NZ
        self.local_ny = LOCAL_NY
        self.local_nx = LOCAL_NX
        self.local_hz = LOCAL_NZ // 2
        self.local_hy = LOCAL_NY // 2
        self.local_hx = LOCAL_NX // 2

    def preprocess(self, combined: torch.Tensor) -> torch.Tensor:
        """
        combined: (N, 2, NZ, NY, NX), channel 0 = OCC, channel 1 = SVS.
        Replicates LocalMap3DDataset.__getitem__ normalisation:
          - OCC clipped to [0, 1]
          - SVS clipped to >= 0, divided by svs_max, clipped to [0, 1].
        """
        out = combined.clone().float()
        out[:, 0] = torch.clamp(out[:, 0], 0.0, 1.0)
        out[:, 1] = torch.clamp(out[:, 1], 0.0, None)
        if self.config.svs_max > 0:
            out[:, 1] = out[:, 1] / self.config.svs_max
        out[:, 1] = torch.clamp(out[:, 1], 0.0, 1.0)
        return out

    def encode(self, combined: torch.Tensor) -> torch.Tensor:
        """
        combined: (N, 2, NZ, NY, NX).
        Returns: (N, latent_dim) — deterministic mean, no sampling (inference_mode).
        """
        with torch.no_grad():
            x = self.preprocess(combined)
            out = self.model.encode(x)
            if isinstance(out, tuple):
                # VAE3D.encode returns (z_sampled, mean, log_var)
                _, mean, _ = out
                return mean
            return out  # plain AE: encode() may return the latent directly

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Optional: reconstruct (N, 2, NZ, NY, NX) for debug viz."""
        with torch.no_grad():
            return torch.sigmoid(self.model.decode(latent))

    def get_latent_dim(self) -> int:
        return self.config.latent_dim

    # ── Build local OCC: body-frame crop of the persistent global_occ_map ─────
    def _build_local_occ_map(self, env_id: int) -> torch.Tensor:
        """
        Binary local OCC in YAW-aligned body frame, cropped from the
        persistent global_occ_map (populated by _update_visit_counts from
        every LiDAR frame). The drone is at the centre, +x forward,
        +y left, +z world-up.
        """
        env = self.env
        cell = self.occ_cell_size
        NZL, NYL, NXL = self.local_nz, self.local_ny, self.local_nx
        NZ, NY, NX = self.occ_map_dims

        # Body-frame cell centres
        xs = (torch.arange(NXL, device=env.device).float() - self.local_hx + 0.5) * cell
        ys = (torch.arange(NYL, device=env.device).float() - self.local_hy + 0.5) * cell
        zs = (torch.arange(NZL, device=env.device).float() - self.local_hz + 0.5) * cell
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        pos_body = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        flat_body = pos_body.reshape(-1, 3)

        # Body → world (yaw + drone pos), then world → warehouse-local
        drone_pos_w = env._robot.data.root_pos_w[env_id]
        drone_quat_w = env._robot.data.root_quat_w[env_id]
        yaw_quat = math_utils.yaw_quat(drone_quat_w.unsqueeze(0)).squeeze(0)
        yaw_exp = yaw_quat.unsqueeze(0).expand(flat_body.shape[0], -1)
        flat_world = quat_apply(yaw_exp, flat_body) + drone_pos_w
        env_origin_xy = env._terrain.env_origins[env_id, :2]
        flat_world[:, :2] -= env_origin_xy

        # Warehouse-local position → global voxel index
        ix = ((flat_world[:, 0] - self.occ_origin[0]) / cell).long()
        iy = ((flat_world[:, 1] - self.occ_origin[1]) / cell).long()
        iz = ((flat_world[:, 2] - self.occ_origin[2]) / cell).long()
        in_bounds = (
            (ix >= 0) & (ix < NX) &
            (iy >= 0) & (iy < NY) &
            (iz >= 0) & (iz < NZ)
        )

        local_flat = torch.zeros((flat_world.shape[0],), dtype=torch.float32, device=env.device)
        if in_bounds.any():
            local_flat[in_bounds] = self.global_occ_map[
                env_id, iz[in_bounds], iy[in_bounds], ix[in_bounds]
            ].float()
        return local_flat.reshape(NZL, NYL, NXL)


    # ── SECTION 2: Build local SVS map in BODY frame ──────────────────────────
    def _build_local_svs_map(
        self, env_id: int, local_occ: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Local SVS, shape (local_nz, local_ny, local_nx), in YAW-aligned body
        frame. The world-frame global_visit_counts grid is inverse-warped
        into the body box: for each body cell we compute its world position,
        find the corresponding world voxel, and read the count there.
        """
        env = self.env
        cell = self.occ_cell_size
        NZL, NYL, NXL = self.local_nz, self.local_ny, self.local_nx
        NZ, NY, NX = self.occ_map_dims

        # ── Body-frame cell centres (z, y, x order matches output layout) ─
        xs = (torch.arange(NXL, device=env.device).float() - self.local_hx + 0.5) * cell
        ys = (torch.arange(NYL, device=env.device).float() - self.local_hy + 0.5) * cell
        zs = (torch.arange(NZL, device=env.device).float() - self.local_hz + 0.5) * cell
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        pos_body = torch.stack([grid_x, grid_y, grid_z], dim=-1)         # (NZL, NYL, NXL, 3)
        flat_body = pos_body.reshape(-1, 3)                              # (M, 3)

        # ── Body → world (apply drone yaw, then translate to drone pos) ───
        drone_pos_w = env._robot.data.root_pos_w[env_id]                 # (3,)
        drone_quat_w = env._robot.data.root_quat_w[env_id]               # (4,)
        yaw_quat = math_utils.yaw_quat(drone_quat_w.unsqueeze(0)).squeeze(0)
        yaw_exp = yaw_quat.unsqueeze(0).expand(flat_body.shape[0], -1)
        env_origin_xy = env._terrain.env_origins[env_id, :2]
        flat_world = quat_apply(yaw_exp, flat_body) + drone_pos_w   # current line
        flat_world[:, :2] -= env_origin_xy                           # add this

        # ── World position → global voxel index ───────────────────────────
        ix = ((flat_world[:, 0] - self.occ_origin[0]) / cell).long()
        iy = ((flat_world[:, 1] - self.occ_origin[1]) / cell).long()
        iz = ((flat_world[:, 2] - self.occ_origin[2]) / cell).long()
        in_bounds = (
            (ix >= 0) & (ix < NX) &
            (iy >= 0) & (iy < NY) &
            (iz >= 0) & (iz < NZ)
        )

        counts_flat = torch.zeros((flat_world.shape[0],), dtype=torch.float32, device=env.device)
        counts_flat[in_bounds] = self.global_visit_counts[
            env_id, iz[in_bounds], iy[in_bounds], ix[in_bounds]
        ]
        counts = counts_flat.reshape(NZL, NYL, NXL)

        # ── SVS = Shannon entropy of normalised visit distribution ────────
        Nt = counts.sum()
        svs = torch.zeros_like(counts)
        if Nt > 0:
            p = counts / Nt
            svs = torch.where(p > 0, -p * torch.log(p), svs)
            if local_occ is None:
                local_occ = self._build_local_occ_map(env_id)
            svs[local_occ > 0.5] = 0.0
        return svs

    # ── SECTION 4: Stack the two maps as separate channels ────────────────────
    def _build_local_combined_map(self, env_id: int) -> torch.Tensor:
        """Returns (2, 8, 16, 16): channel 0 = OCC, channel 1 = SVS."""
        local_occ = self._build_local_occ_map(env_id)
        local_svs = self._build_local_svs_map(env_id, local_occ=local_occ)
        return torch.stack([local_occ, local_svs], dim=0)

    # ── Update both buffers from per-step sensor data ─────────────────────────
    def _update_visit_counts(self, env_ids: torch.Tensor | None = None):
        """
        Every step:
          (a) Increment global_visit_counts at the drone's warehouse-local voxel.
          (b) Mark global_occ_map = 1 at every voxel that received a finite
              LiDAR hit this frame (warehouse-local coords).
        Both buffers are persistent across steps and reset only in _reset_idx.
        """
        env = self.env
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        env_origins = env.scene.env_origins[env_ids]                # (E, 3)
        NZ, NY, NX = self.occ_map_dims
        cell = self.occ_cell_size

        # ─── (a) Visit counts at the drone's voxel ────────────────────────
        pos_w = env._robot.data.root_pos_w[env_ids]                 # (E, 3)
        local_pos = pos_w.clone()
        local_pos[:, :2] -= env_origins[:, :2]                      # warehouse-local

        iz = ((local_pos[:, 2] - self.occ_origin[2]) / cell).long()
        iy = ((local_pos[:, 1] - self.occ_origin[1]) / cell).long()
        ix = ((local_pos[:, 0] - self.occ_origin[0]) / cell).long()
        valid = (iz >= 0) & (iz < NZ) & (iy >= 0) & (iy < NY) & (ix >= 0) & (ix < NX)
        if valid.any():
            ve = env_ids[valid]
            self.global_visit_counts[ve, iz[valid], iy[valid], ix[valid]] += 1.0

        # ─── (b) OCC from LiDAR hits, accumulated across steps ────────────
        # ray_hits_w shape: (num_envs, B, 3) in world frame; inf for no-return.
        hits_w = env.ray_caster.data.ray_hits_w[env_ids]            # (E, B, 3)
        hits_valid = torch.isfinite(hits_w).all(dim=-1)             # (E, B)

        # World → warehouse-local for the x/y axes (z is unchanged because
        # env_origins is only offset in the horizontal plane).
        hits_local = hits_w.clone()
        hits_local[..., :2] -= env_origins.unsqueeze(1)[..., :2]    # broadcast (E, 1, 2)

        hix = ((hits_local[..., 0] - self.occ_origin[0]) / cell).long()
        hiy = ((hits_local[..., 1] - self.occ_origin[1]) / cell).long()
        hiz = ((hits_local[..., 2] - self.occ_origin[2]) / cell).long()
        hits_in_bounds = (
            (hix >= 0) & (hix < NX) &
            (hiy >= 0) & (hiy < NY) &
            (hiz >= 0) & (hiz < NZ)
        )
        write = hits_valid & hits_in_bounds                         # (E, B)

        if write.any():
            B = hits_w.shape[1]
            env_idx_grid = env_ids.view(-1, 1).expand(-1, B)        # (E, B)
            self.global_occ_map[
                env_idx_grid[write], hiz[write], hiy[write], hix[write]
            ] = 1

    def _save_local_maps(self, env_ids: torch.Tensor):                                                                                                                   
        """Save combined (2, 8, 16, 16) maps for alive envs to disk."""
        step = self.common_step_counter                                                                                                                                  
        for env_id in env_ids.tolist():
            combined = self._build_local_combined_map(env_id)                                                                                                            
            path = os.path.join(LOCAL_MAPS_SAVE_DIR, f"env{env_id}_step{step}.npy")                                                                                      
            np.save(path, combined.cpu().numpy()) 


class UavNavigationEnv(DirectRLEnv):
    cfg: UavNavigationEnvCfg

    def __init__(self, cfg: UavNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.low_level_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_yaw_cmd = torch.zeros(self.num_envs, 1, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.rel_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

        # consecutive steps each env has been alive (gates map saving)
        self.alive_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  

        self.x_min = -28.0
        self.x_max = 8.0
        self.y_min = -41.4
        self.y_max = 33.42
        self.z_min = 0.0
        self.z_max = 9.30
        
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "life",
                "died",
                "time_out",
                "action_reg_diff",
                "final_distance_to_goal",
            ]
        }

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        #self.set_debug_vis(self.cfg.debug_vis)

        self.distance_to_bounds_x = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_bounds_y = torch.zeros(self.num_envs, device=self.device)
        self.final_distance_to_goal_b = torch.zeros(self.num_envs, device=self.device)

        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0

        self.policy_network = self._load_policy_network()
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        self.vae_encoder = VAEImageEncoder(vae_config, device=self.device)
        self.max_depth = 10.0  
        # 3D occupancy / SVS map builder — owns global_occ_map & global_visit_counts
        self.autoencoder3D = autoencoder_3d(self)

        os.makedirs(LOCAL_MAPS_SAVE_DIR, exist_ok=True)
        print(f"[LocalMap] Online OCC: shape={self.autoencoder3D.occ_map_dims}, "
              f"cell={self.autoencoder3D.occ_cell_size} m")


    def _load_policy_network(self):
        checkpoint = torch.load(
            "/workspace/environment/quadcopter_vel_control/projects/quadcopter_vel_control/logs/skrl/quadcopter_vel_control_direct/2026-04-08_08-45-39_Manual_PPO/26-04-08_08-45-43-335682_PPO/checkpoints/best_agent.pt",
            map_location=self.device
        )

        preprocessor = checkpoint["state_preprocessor"]
        self.ll_running_mean     = preprocessor["running_mean"].float().to(self.device)
        self.ll_running_variance = preprocessor["running_variance"].float().to(self.device)
        self.ll_epsilon          = 1e-8
        self.ll_clip_threshold   = 5.0

        policy = PolicyNet()
        missing, unexpected = policy.load_state_dict(checkpoint["policy"], strict=False)
        assert "net.0.weight" not in missing, "Core weights failed to load!"
        policy.to(self.device)
        policy.eval()
        return policy

    def _normalize_ll_obs(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.ll_running_mean) / torch.sqrt(self.ll_running_variance + self.ll_epsilon)
        return torch.clamp(normalized, -self.ll_clip_threshold, self.ll_clip_threshold)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.camera = Camera(self.cfg.camera)
        self.scene.sensors["camera"] = self.camera
        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # ── Enumerate warehouse mesh prims for the OCC raycaster ──────────
        # Same enumeration pattern as vae_container/Vae/occupancymap3d_fw.py,
        # rooted at /World/ground (where TerrainImporterCfg places the
        # full_warehouse USD).

        stage = omni.usd.get_context().get_stage()
        mesh_paths = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith("/World/ground") and prim.IsA(UsdGeom.Mesh):
                mesh_paths.append(path)
        print(f"[RayCaster] {len(mesh_paths)} warehouse meshes registered for occupancy raycaster")
        self.cfg.ray_caster.mesh_prim_paths = mesh_paths
        self.ray_caster = MultiMeshRayCaster(self.cfg.ray_caster)
        self.scene.sensors["ray_caster"] = self.ray_caster

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self.target_vel_cmd[:, :3] = self._actions[:, :3] * self.cfg.max_velocity
        self.target_yaw_cmd = self._actions[:, 3:4] * self.cfg.max_yaw_rate

    def _apply_action(self):
        if self.common_step_counter % self.cfg.decimation_low_level == 0:
            low_level_obs = torch.hstack((
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self.target_vel_cmd,
                self.target_yaw_cmd.reshape(-1, 1),
            ))
            low_level_obs = low_level_obs.to(self.device).to(torch.float32)
            with torch.no_grad():
                obs_norm = self._normalize_ll_obs(low_level_obs)
                self.low_level_actions = self.policy_network(obs_norm)

            self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.low_level_actions[:, 0] + 1.0) / 2.0
            self._moment[:, 0, :] = self.cfg.moment_scale * self.low_level_actions[:, 1:]

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        # ── Update visit counts at every step (all envs) ──────────────────────
        self.autoencoder3D._update_visit_counts()

        # ── Save combined maps periodically ───────────────────────────────────
        origins = self.scene.env_origins                                                                                                                                     
        pos_w   = self._robot.data.root_pos_w
        local_x = pos_w[:, 0] - origins[:, 0]                                                                                                                                
        local_y = pos_w[:, 1] - origins[:, 1]                                                                                                                                
        is_alive = (                                                                                                                                                         
            (local_x > self.x_min) & (local_x < self.x_max) &                                                                                                                
            (local_y > self.y_min) & (local_y < self.y_max) &
            (pos_w[:, 2] > self.z_min) & (pos_w[:, 2] < self.z_max)                                                                                                          
        )                                                          
        self.alive_steps[is_alive]  += 1                                                                                                                                     
        self.alive_steps[~is_alive] = 0 
                                                                                                                                                                            
        # ── Save maps periodically — only for envs alive long enough ──────────
        if LOCAL_MAP_SAVE_EVERY > 0 and self.common_step_counter >= LOCAL_MAP_START_STEP and self.common_step_counter % LOCAL_MAP_SAVE_EVERY == 0:                                                                               
            eligible = (self.alive_steps >= MIN_ALIVE_STEPS_TO_SAVE).nonzero(as_tuple=False).view(-1)
            if eligible.numel() > 0:                                                                                                                                         
                self.autoencoder3D._save_local_maps(eligible) 
                # ── Periodic sanity plot of the per-env online OCC (env 0) ────────────
        if ONLINE_OCC_VIZ_EVERY > 0 and self.common_step_counter > 0 \
                and self.common_step_counter % ONLINE_OCC_VIZ_EVERY == 0:
            self.autoencoder3D._save_online_occ_viz(env_id=0)

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

        # if self.common_step_counter % 20 == 0:
        #     depth_vis = depth[0, :, :, 0] / self.max_depth
        #     depth_np = depth_vis.detach().cpu().numpy()
        #     plt.imshow(depth_np, cmap='plasma', vmin=0, vmax=1)
        #     plt.colorbar(label='Depth (m)')
        #     plt.title('Raw Depth')
        #     plt.savefig('depth_check.png')
        #     plt.close()

        #     collision_np = collision[0, 0].detach().cpu().numpy()
        #     plt.imshow(collision_np, cmap='plasma', vmin=0, vmax=1)
        #     plt.title('Collision Image')
        #     plt.savefig('collision_check.png')
        #     plt.close()

        #     recon = self.vae_encoder.decode(latent)
        #     recon_np = recon[0, 0].detach().cpu().numpy()
        #     plt.imshow(recon_np, cmap='plasma', vmin=0, vmax=1)
        #     plt.title('Reconstruction')
        #     plt.savefig('recon_check.png')
        #     plt.close()

        obs = torch.cat(
            [
                self.rel_pos_b,
                self._prev_actions,
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
        # print("occ_origin:", self.occ_origin)
        # print("env_origin[0]:", self._terrain.env_origins[0])
        # print("env_origin[1]:", self._terrain.env_origins[1])
        # print("env_origin[2]:", self._terrain.env_origins[2])
        origins = self.scene.env_origins
        lin_vel_sum = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sum = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        lin_vel = 1 - torch.tanh(lin_vel_sum / 0.8)
        ang_vel = 1 - torch.tanh(ang_vel_sum / 0.8)
        self._episode_sums["final_distance_to_goal"] += self.final_distance_to_goal * self.step_dt
        distance_to_goal_mapped = 1 - torch.tanh(self.final_distance_to_goal / 0.8)

        #pos_local = self._robot.data.root_pos_w - origins[:, 0]  # (num_envs, 3)
        self.distance_to_bounds_x = ((self._robot.data.root_pos_w[:, 0] - origins[:, 0] > self.x_min)) & ((self.x_max > self._robot.data.root_pos_w[:, 0] - origins[:, 0]))
        self.distance_to_bounds_y = ((self._robot.data.root_pos_w[:, 1] - origins[:, 1] > self.y_min)) & ((self.y_max > self._robot.data.root_pos_w[:, 1] - origins[:, 1]))
        # square_side = self.arena_size
        # relative_distance_to_bounds_x = torch.abs((origins[:, 0]) - self._robot.data.root_pos_w[:, 0])
        # relative_distance_to_bounds_y = torch.abs((origins[:, 1]) - self._robot.data.root_pos_w[:, 1])
        # self.distance_to_bounds_x = (square_side / 2) - relative_distance_to_bounds_x
        # self.distance_to_bounds_y = (square_side / 2) - relative_distance_to_bounds_y

        action_diff = self._actions - self._prev_actions
        action_reg_diff = torch.norm(action_diff, p=2, dim=-1)
        action_reg_diff = 1 - torch.tanh(action_reg_diff / 0.8)

        is_alive_bounds = torch.logical_and(self.distance_to_bounds_x, self.distance_to_bounds_y)
        is_alive_height = torch.logical_and(self._robot.data.root_pos_w[:, 2] > self.z_min, self._robot.data.root_pos_w[:, 2] < self.z_max)
        is_alive = torch.logical_and(is_alive_bounds, is_alive_height)
        life = torch.where(is_alive, self.cfg.alive_reward_scale, self.cfg.death_reward_scale)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_reg_diff": action_reg_diff * self.cfg.rew_scale_action_reg * self.step_dt,
            "life": life * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = (self._robot.data.root_pos_w[:, 2] < self.z_min) | \
               (self._robot.data.root_pos_w[:, 2] > self.z_max) | \
               ( ~self.distance_to_bounds_x.bool()) | \
               ( ~self.distance_to_bounds_y.bool())
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        num_deaths = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        num_timeouts = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self._episode_sums["died"] += num_deaths
        self._episode_sums["time_out"] += num_timeouts

        self.extras["log"] = dict()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key in ["died", "time_out"]:
                extras["Episode_Termination/" + key] = episodic_sum_avg / self.max_episode_length_s
            elif key in ["final_distance_to_goal"]:
                extras["Episode_Info/" + key] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"].update(extras)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._prev_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self.alive_steps[env_ids] = 0.0


        n = len(env_ids)
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(
            self._desired_pos_w[env_ids, 0]
        ).uniform_(self.x_min + 1.0, self.x_max - 1.0)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(
            self._desired_pos_w[env_ids, 1]
        ).uniform_(self.y_min + 1.0, self.y_max - 1.0)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(self.z_min + 1.0, self.z_max - 1.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # default_root_state[:, 2] = 1.0
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 0] = torch.zeros_like(default_root_state[:, 0]).uniform_(self.x_min + 1.0, self.x_max - 1.0)
        #self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(self.y_min + 1.0, self.y_max - 1.0)
        default_root_state[:, 1] = torch.zeros_like(default_root_state[:, 1]).uniform_(0.0, self.y_max - 1.0)
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(self.z_min + 1.0, self.z_max - 1.0)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Reset visit counts for terminated envs ────────────────────────────
        self.autoencoder3D.global_visit_counts[env_ids] = 0.0
        # ── Reset OCC buffer for terminated envs ──────────────────────────────
        self.autoencoder3D.global_occ_map[env_ids] = 0


    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass