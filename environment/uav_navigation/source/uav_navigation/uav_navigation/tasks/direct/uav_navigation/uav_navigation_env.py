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
from .autoencoder3D import VAE3D
import inspect
from tqdm import tqdm
import cv2
import omni.usd
from pxr import UsdGeom, Usd

wandb.login()

project = "uav_navigation"

LOCAL_NZ = 8                  # local map depth  (z axis)
LOCAL_NY = 16                 # local map height (y axis)
LOCAL_NX = 16                 # local map width  (x axis)
LOCAL_CELL_SIZE = 0.25

# Policy Architecture per Velocity controller
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

# Classi di configurazione Vae e 3D-AE con latent_dim, path to checkpoint, e altre informazioni utili

class vae_config:
    use_vae = True
    latent_dims = 512
    #032824
    model_file = (
        "/workspace/vae_container/Vae/runs/l7dd58vd/mild-sweep-1/checkpoints/vae_best_20260606_063918.pt"
    )
    model_folder = "/workspace/vae_container/Vae/checkpoint"
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = False

class ae3d_config:
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
    svs_max            = 0.243812

# Classe per VAE. 
class VAEImageEncoder:

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device
        self._max_depth_val = 10.0
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
        ).to(device)
        # combine module path with model file name
        weight_file_path = self.config.model_file
        state_dict = self.clean_state_dict(torch.load(weight_file_path, map_location=self.device))
        # for k, v in state_dict.items():
        #     print(f"{k}: {v.shape}")
        missing, unexpected = self.vae_model.load_state_dict(state_dict, strict=False)
        core_keys = [k for k in missing if "conv" in k or "dense0" in k]
        if core_keys:
            raise RuntimeError(f"Core architecture mismatch: {core_keys}")
        self.vae_model.eval()

    def clean_state_dict(self, state_dict):
        clean_dict = {}
        for key, value in state_dict.items():
            if "module." in key:
                key = key.replace("module.", "")
            if "dronet." in key:
                key = key.replace("dronet.", "encoder.")
            clean_dict[key] = value
        return clean_dict

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():

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
            z_sampled, means, log_var = self.vae_model.encode(interpolated_image)
            # n_clipped = ((log_var < -10) | (log_var > 4)).sum().item()
            # if n_clipped > 0:
            #     print(f"WARNING: {n_clipped} log_var values were clamped")

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


    def preprocess_depth(self, depth_torch: torch.Tensor) -> torch.Tensor:
        """
        Prepare a depth tensor for the depth-input VAE.

        Mirrors the offline training preprocessing exactly:
          - nan / +inf / -inf → 0.0           (matches np.nan_to_num used in training)
          - clamp to [0, MAX_DEPTH]            (matches np.clip)
          - normalise to [0, 1] by /MAX_DEPTH  (matches the division by MAX_DEPTH)

        Input  : (N, H, W, 1) on self.device, float, raw camera distance_to_camera
        Output : (N, 1, H, W) on self.device, in [0, 1]
        """
        # Controllo la forma del tensore di ingresso. Dopo l'if deve essere (N, H, W). 
        if depth_torch.ndim == 4 and depth_torch.shape[-1] == 1:
            depth = depth_torch.squeeze(-1)                  # (N, H, W)
        else:
            depth = depth_torch

        # Adesso posso normalizzare i valori con la stessa logica del training.
        depth = torch.nan_to_num(depth.float(), nan=self._max_depth_val,
                                 posinf=self._max_depth_val, neginf=self._max_depth_val)

        depth = torch.clamp(depth, 0.0, self._max_depth_val)
        depth = depth / self._max_depth_val                  # (N, H, W) in [0, 1]
        depth = depth.unsqueeze(1)                           # (N, 1, H, W)
        
        # Controllo che H e W siano quelli attesi.
        if depth.shape[-2:] != tuple(self.config.image_res):
            depth = torch.nn.functional.interpolate(
                depth,
                size=self.config.image_res,
                mode=self.config.interpolation_mode,
            )
        return depth

# Classe per 3D autoencoder.

class autoencoder_3d:
    def __init__(self, env, config=ae3d_config):
        self.env = env
        self.config = config
        self.device = env.device

        # ── Geometry of the global grids ───────────────────────────────────
        self.occ_cell_size = float(env.cfg.occ_cell_size)
        self.occ_origin = torch.tensor(
            [env.x_min, env.y_min, env.z_min],
            dtype=torch.float32, device=self.device,
        )
        # Sto dividendo l'ambiente in celle
        self.NX = int(math.ceil((env.x_max - env.x_min) / self.occ_cell_size))
        self.NY = int(math.ceil((env.y_max - env.y_min) / self.occ_cell_size))
        self.NZ = int(math.ceil((env.z_max - env.z_min) / self.occ_cell_size))
        self.occ_map_dims = (self.NZ, self.NY, self.NX)

        # ── Persistent per-env buffers, online from sensors ────────────────
        self.global_occ_map = torch.zeros(
            (env.num_envs, self.NZ, self.NY, self.NX),
            dtype=torch.uint8, device=self.device,
        )
        self.global_visit_counts = torch.zeros(
            (env.num_envs, self.NZ, self.NY, self.NX),
            dtype=torch.float32, device=self.device,
        )
        self.entered_new_cell = torch.zeros(
            env.num_envs, dtype=torch.bool, device=self.device,
        )

        # ── Local map shape ────────────────────────────────────────────────
        # Larghezza
        self.local_nz = LOCAL_NZ
        self.local_ny = LOCAL_NY
        self.local_nx = LOCAL_NX
        # Altezza
        self.local_hz = LOCAL_NZ // 2 
        self.local_hy = LOCAL_NY // 2
        self.local_hx = LOCAL_NX // 2

        # ── Trained 3D AE ──────────────────────────────────────────────────
        self.model = VAE3D(
            input_dim         = 2,
            latent_dim        = config.latent_dim,
            with_logits       = True, # Usato solo su decoder, quindi ininfluente.
            inference_mode    = True,
            num_conv_layers   = config.num_conv_layers,
            use_residual      = config.use_residual,
            residual_every    = config.residual_every,
            num_deconv_layers = config.num_deconv_layers,
            use_skip          = config.use_skip,
        ).to(self.device)

        print(f"[AE3D] Loading weights from {config.model_file}")
        state_dict = torch.load(config.model_file, map_location=self.device)
        clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        #print(f"[AE3D] Missing keys:    {missing}")
        #print(f"[AE3D] Unexpected keys: {unexpected}")
        core = [k for k in missing if "conv" in k or "dense" in k]
        if core:
            raise RuntimeError(f"3D AE state_dict mismatch on core layers: {core}")
        self.model.eval()


    # ── AE3D interface (preprocess / encode / decode / latent dim) ────────────
    def preprocess(self, combined: torch.Tensor) -> torch.Tensor:
        out = combined.clone().float()
        out[:, 0] = torch.clamp(out[:, 0], 0.0, 1.0)
        out[:, 1] = torch.clamp(out[:, 1], 0.0, None)
        # Anche se nel trainig svs_max = 1.0 , se non è mai stato raggiunto allora devo normalizzare.
        # self.config.svs_max è stato preso dalla simulazione.
        if self.config.svs_max > 0:
            out[:, 1] = out[:, 1] / self.config.svs_max
        out[:, 1] = torch.clamp(out[:, 1], 0.0, 1.0)
        return out

    def encode(self, combined: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.preprocess(combined)
            out = self.model.encode(x)
            if isinstance(out, tuple):
                _, mean, _ = out
                return mean
            return out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.model.decode(latent))


    def build_local_features_batched(self) -> dict:
        """
        Build local SVS + OCC maps. Moreover, compute the distance-to-obstacle reward and the visit counts
        Returns:
          'combined':         (N, 2, NZL, NYL, NXL)  channel 0 OCC, channel 1 SVS
          'local_occ':        (N, NZL, NYL, NXL)    binary
          'dist_to_obstacle': (N,)                  metres
          'Nt':               (N,)                  sum of visit counts
        """
        env = self.env
        cell = self.occ_cell_size
        NZL, NYL, NXL = self.local_nz, self.local_ny, self.local_nx     # numero di celle nella singla cella
        NZ, NY, NX = self.occ_map_dims                                  # numero di celle nell'env
        N = env.num_envs

        # Body-frame grid (shared across envs)
        # Corrispondente in metri della cella a partire dla drone in pratica va da -2 a +2 in larghezza e da -1 a +1 in altezza
        xs = (torch.arange(NXL, device=env.device).float() - self.local_hx + 0.5) * cell 
        ys = (torch.arange(NYL, device=env.device).float() - self.local_hy + 0.5) * cell
        zs = (torch.arange(NZL, device=env.device).float() - self.local_hz + 0.5) * cell
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
        flat_body = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)   # (M, 3)
        M = flat_body.shape[0] # numero totale di voxel ->  M = 8 × 16 × 16 = 2048

        # Per-env transforms
        drone_pos_w = env._robot.data.root_pos_w                                   # (N, 3)
        drone_quat_w = env._robot.data.root_quat_w                                 # (N, 4)
        yaw_quat = math_utils.yaw_quat(drone_quat_w)                               # (N, 4)

        yaw_flat = yaw_quat.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)          # (N*M, 4)
        body_flat = flat_body.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)        # (N*M, 3)
        # Le mappe sono allineate in yaw poi aggiungo le coordinate del drone in world per ottenere le coordinate globali dei voxel.
        world_flat = quat_apply(yaw_flat, body_flat).reshape(N, M, 3)              # (N, M, 3)
        world_flat = world_flat + drone_pos_w.unsqueeze(1)

        # Shared-warehouse: index global grids by world coordinates directly.
        # processo inverso, passo da metri a indice di cella
        ix = ((world_flat[:, :, 0] - self.occ_origin[0]) / cell).long() 
        iy = ((world_flat[:, :, 1] - self.occ_origin[1]) / cell).long()
        iz = ((world_flat[:, :, 2] - self.occ_origin[2]) / cell).long()

        in_bounds = (
            (ix >= 0) & (ix < NX) &
            (iy >= 0) & (iy < NY) &
            (iz >= 0) & (iz < NZ)
        )                                                                          # (N, M)

        ix_c = ix.clamp(0, NX - 1)
        iy_c = iy.clamp(0, NY - 1)
        iz_c = iz.clamp(0, NZ - 1)

        env_idx = torch.arange(N, device=env.device).unsqueeze(1).expand(-1, M)    # (N, M)

        # Alla fine local_occ è una mappa 8x16x16 con 1 se c'è l'ostacolo e 0 altrimenti
        occ_flat = self.global_occ_map[env_idx, iz_c, iy_c, ix_c].float()
        occ_flat = occ_flat * in_bounds.float()
        local_occ = occ_flat.reshape(N, NZL, NYL, NXL)

        counts_flat = self.global_visit_counts[env_idx, iz_c, iy_c, ix_c]
        counts_flat = counts_flat * in_bounds.float()
        counts = counts_flat.reshape(N, NZL, NYL, NXL)

        Nt = counts.flatten(1).sum(dim=1)                                          # (N,)
        Nt_safe = Nt.clamp(min=1e-8).view(N, 1, 1, 1)
        p = counts / Nt_safe
        svs = torch.where(p > 0, -p * torch.log(p), torch.zeros_like(p))
        svs = svs * (local_occ < 0.5).float()
        svs = svs * (Nt > 0).float().view(N, 1, 1, 1)

        # Distance from body origin (0,0,0) to any occupied voxel centre
        dists_body = torch.norm(flat_body, dim=1)                                  # (M,)
        max_dist = float(self.local_hx) * cell
        occ_mask = (occ_flat > 0.5)                                                # (N, M)
        dists_exp = dists_body.unsqueeze(0).expand(N, -1)
        dists_masked = torch.where(
            occ_mask, dists_exp,
            torch.full_like(dists_exp, float("inf")),
        )
        dist_to_obstacle = dists_masked.min(dim=1).values
        dist_to_obstacle = torch.where(
            torch.isinf(dist_to_obstacle),
            torch.full_like(dist_to_obstacle, max_dist),
            dist_to_obstacle,
        )

        combined = torch.stack([local_occ, svs], dim=1)                            # (N, 2, NZL, NYL, NXL)

        return {
            "combined":         combined,
            "local_occ":        local_occ,
            "dist_to_obstacle": dist_to_obstacle,
            "Nt":               Nt,
        }

    # ── Per-step buffer updates from sensors ──────────────────────────────────
    def _update_visit_counts(self, env_ids: torch.Tensor | None = None):
        env = self.env
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        NZ, NY, NX = self.occ_map_dims
        cell = self.occ_cell_size

        self.entered_new_cell.zero_()

        pos_w = env._robot.data.root_pos_w[env_ids]

        iz = ((pos_w[:, 2] - self.occ_origin[2]) / cell).long()
        iy = ((pos_w[:, 1] - self.occ_origin[1]) / cell).long()
        ix = ((pos_w[:, 0] - self.occ_origin[0]) / cell).long()
        valid = (iz >= 0) & (iz < NZ) & (iy >= 0) & (iy < NY) & (ix >= 0) & (ix < NX)
        if valid.any():
            ve = env_ids[valid]
            izv, iyv, ixv = iz[valid], iy[valid], ix[valid]
            # Sample the counts BEFORE incrementing: a count of 0 means this
            # voxel is being entered for the first time.
            counts_before = self.global_visit_counts[ve, izv, iyv, ixv]
            is_first_visit = counts_before == 0
            self.global_visit_counts[ve, izv, iyv, ixv] += 1.0
            # Mark only the envs that actually entered a fresh voxel.
            self.entered_new_cell[ve[is_first_visit]] = True

        hits_w = env.ray_caster.data.ray_hits_w[env_ids]
        hits_valid = torch.isfinite(hits_w).all(dim=-1)
        hix = ((hits_w[..., 0] - self.occ_origin[0]) / cell).long()
        hiy = ((hits_w[..., 1] - self.occ_origin[1]) / cell).long()
        hiz = ((hits_w[..., 2] - self.occ_origin[2]) / cell).long()
        hits_in_bounds = (
            (hix >= 0) & (hix < NX) &
            (hiy >= 0) & (hiy < NY) &
            (hiz >= 0) & (hiz < NZ)
        )
        write = hits_valid & hits_in_bounds
        if write.any():
            B = hits_w.shape[1]
            env_idx_grid = env_ids.view(-1, 1).expand(-1, B)
            self.global_occ_map[
                env_idx_grid[write], hiz[write], hiy[write], hix[write]
            ] = 1

# Classe che descrive l'environment con la struttura di sempre.

class UavNavigationEnv(DirectRLEnv):
    cfg: UavNavigationEnvCfg

    def __init__(self, cfg: UavNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print(f"[UavNavigationEnv] self.device = {self.device}")
        print(f"[UavNavigationEnv] self.sim.cfg.device = {self.sim.cfg.device}")

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
        self.y_min = -41.40
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
                "distance_to_obstacles",
                "exploration",
                "success",
                "action_reg_diff",
                "final_distance_to_goal",
                "mean_dist_to_obstacle",   
                "cells_visited",           
            ]
        }

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.distance_to_bounds_x = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_bounds_y = torch.zeros(self.num_envs, device=self.device)
        self.final_distance_to_goal_b = torch.zeros(self.num_envs, device=self.device)

        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0

        self.policy_network = self._load_policy_network()
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        # Curriculum state (discrete levels: each entry is (frac, center_x, center_y)).
        # center=None means "map center" -> combined with frac=1.0 this spans the full map.
        self._curriculum_schedule = [
            (0.2, -10.0, -23.0),   # level 1
            (0.3, -10.0, -4.0),   # level 2
            (0.4, -10.0, -4.0),   # level 3
            (0.5, -10.0, -4.0),   # level 4
            (1.0, None, None),   # level 5  (full map)
            (0.2, -10.0,  16.0),   # level 6
            (0.3, -10.0,  16.0),   # level 7
            (0.4, -10.0,  16.0),   # level 8
            (0.5, -10.0,  16.0),   # level 9
        ]
        self._curr_level = 0
        self._curr_episodes = 0
        self._curr_successes = 0

        
        # True if the drone entered the goal sphere at any point this episode.
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Cached perception features used by _get_rewards / _get_dones ─────
        # Default to a large value so step 0 doesn't false-positive a collision.
        _local_half_extent = (LOCAL_NX // 2) * LOCAL_CELL_SIZE
        # La distanza al primo ostacolo è già massima, altrimenti sarebbe già in collisione.
        self._dist_to_obstacle_cache = torch.full(
            (self.num_envs,), _local_half_extent, device=self.device,
        )
        self._Nt_cache = torch.zeros(self.num_envs, device=self.device)

        self.vae_encoder = VAEImageEncoder(vae_config, device=self.device)
        self.max_depth = self.vae_encoder._max_depth_val 
        # Minimo valore di profondità 
        self._min_depth_cache = torch.full(
            (self.num_envs,),
            float(self.cfg.camera.spawn.clipping_range[1]),
            device=self.device,
        )
        # 3D occupancy / SVS map builder — owns global_occ_map & global_visit_counts
        self.autoencoder3D = autoencoder_3d(self)

        # Static clearance field + free-cell list used to spawn collision-free.
        self._build_spawn_clearance_field()

        self._success_thresholds = torch.tensor(self.cfg.goal_success_thresholds, device=self.device)
        self._success_claimed = torch.zeros(
            self.num_envs, len(self.cfg.goal_success_thresholds), dtype=torch.bool, device=self.device
        )


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

    def _build_spawn_clearance_field(self):
        """Load the static warehouse occupancy map, downsample it to the env's
        occ grid, and precompute (i) a metric clearance field to the nearest
        OCCUPIED cell and (ii) the world centres of all cells that are both
        clear (>= spawn_clearance) and inside the spawn band / inset bounds.
        These are used to spawn drone and goal only in collision-free space."""
        from scipy import ndimage

        meta = np.load(self.cfg.occupancy_meta_path, allow_pickle=True).item()
        src_cell = float(meta["cell_size"])                 # 0.05 m
        cell = float(self.cfg.occ_cell_size)                # 0.25 m
        f = int(round(cell / src_cell))                     # 5

        occ_full = np.load(self.cfg.occupancy_map_path)     # (NZf, NYf, NXf) uint8, [z,y,x]
        occ = (occ_full == 2)                               # only OCCUPIED counts as obstacle
        NZf, NYf, NXf = occ.shape
        NZ = -(-NZf // f); NY = -(-NYf // f); NX = -(-NXf // f)   # ceil division
        occ = np.pad(occ, [(0, NZ * f - NZf), (0, NY * f - NYf), (0, NX * f - NXf)],
                     mode="constant", constant_values=False)
        occ_ds = occ.reshape(NZ, f, NY, f, NX, f).any(axis=(1, 3, 5))   # (NZ, NY, NX)

        clearance = ndimage.distance_transform_edt(~occ_ds) * cell      # metres, [z,y,x]
        field = torch.from_numpy(clearance).to(self.device).float()

        zc = self.z_min + (torch.arange(NZ, device=self.device).float() + 0.5) * cell
        yc = self.y_min + (torch.arange(NY, device=self.device).float() + 0.5) * cell
        xc = self.x_min + (torch.arange(NX, device=self.device).float() + 0.5) * cell
        gz, gy, gx = torch.meshgrid(zc, yc, xc, indexing="ij")

        valid = (
            (field >= self.cfg.spawn_clearance)
            & (gz >= self.cfg.spawn_z_min) & (gz <= self.cfg.spawn_z_max)
            & (gx >= self.x_min + 1.0) & (gx <= self.x_max - 1.0)
            & (gy >= self.y_min + 1.0) & (gy <= self.y_max - 1.0)
        )
        self._free_xyz = torch.stack([gx[valid], gy[valid], gz[valid]], dim=1)   # (M, 3)
        if self._free_xyz.shape[0] == 0:
            raise RuntimeError(
                "No free spawn cells found; check occupancy map path or lower spawn_clearance."
            )
        # print(f"[Spawn] {self._free_xyz.shape[0]} free spawn cells "
        #       f"(clearance >= {self.cfg.spawn_clearance} m)")


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
        
    def _sample_drone_and_goal(self, n: int):
        frac, cx_sched, cy_sched = self._curriculum_schedule[self._curr_level]

        cx = cx_sched if cx_sched is not None else 0.5 * (self.x_min + self.x_max)
        cy = cy_sched if cy_sched is not None else 0.5 * (self.y_min + self.y_max)

        near_spawn = self._curr_level < self.cfg.curriculum_near_levels

        def axis(lo, hi, center):
            h = 0.5 * (hi - lo) * frac
            return torch.empty(n, device=self.device).uniform_(center - h, center + h)

        def fixed_axis(lo, hi):
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        def sample_box():
            return torch.stack(
                [axis(self.x_min + 1.0, self.x_max - 1.0, cx),
                 axis(self.y_min + 1.0, self.y_max - 1.0, cy),
                 fixed_axis(self.cfg.spawn_z_min, self.cfg.spawn_z_max)],
                dim=1,
            )

        drone = sample_box()

        if self.cfg.eval_fixed_spawn_xy is not None:
            drone[:, 0] = self.cfg.eval_fixed_spawn_xy[0]
            drone[:, 1] = self.cfg.eval_fixed_spawn_xy[1]
            drone[:, 2] = 0.5 * (self.cfg.spawn_z_min + self.cfg.spawn_z_max)

        if near_spawn:
            R = self.cfg.curriculum_near_radius
            direction = torch.randn(n, 3, device=self.device)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
            magnitude = torch.empty(n, device=self.device).uniform_(self.cfg.min_goal_separation, R)
            goal = drone + direction * magnitude.unsqueeze(1)
            goal[:, 0] = goal[:, 0].clamp(self.x_min + 1.0, self.x_max - 1.0)
            goal[:, 1] = goal[:, 1].clamp(self.y_min + 1.0, self.y_max - 1.0)
            goal[:, 2] = goal[:, 2].clamp(self.cfg.spawn_z_min, self.cfg.spawn_z_max)
            return drone, goal

        # Later levels: independent goal in the box, with a minimum separation.
        goal = drone.clone()
        too_close = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(self.cfg.curriculum_max_resample):
            if not too_close.any():
                break
            cand = sample_box()
            goal[too_close] = cand[too_close]
            dist = torch.linalg.norm(goal - drone, dim=1)
            too_close = dist < self.cfg.min_goal_separation
        return drone, goal



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
        
        pos_w = self._robot.data.root_pos_w

        # if self.common_step_counter < 3:
        #     print(f"[SPAWN] actual root_pos_w[0] = ("
        #           f"{pos_w[0,0]:.2f}, {pos_w[0,1]:.2f}, {pos_w[0,2]:.2f})")

        self.distance_to_bounds_x = (pos_w[:, 0] > self.x_min + 0.3) & (pos_w[:, 0] < self.x_max - 0.3)
        self.distance_to_bounds_y = (pos_w[:, 1] > self.y_min + 0.3) & (pos_w[:, 1] < self.y_max - 0.3)

        is_alive = (
            self.distance_to_bounds_x &
            self.distance_to_bounds_y &
            (pos_w[:, 2] > self.z_min + 0.3) & (pos_w[:, 2] < self.z_max - 0.3)
        )
        self.alive_steps[is_alive]  += 1
        self.alive_steps[~is_alive] = 0

        self.rel_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w
        )
        self.final_distance_to_goal = torch.linalg.norm(self.rel_pos_b, dim=1)
        
        self._reached_goal |= self.final_distance_to_goal < self.cfg.goal_radius
        
        depth = self.camera.data.output["distance_to_camera"]  # (N, H, W, 1)
        # ── Min depth in the camera FOV — used by distance-to-obstacles reward ─
        max_range = float(self.cfg.camera.spawn.clipping_range[1])
        depth_clean = torch.nan_to_num(depth, nan=max_range, posinf=max_range, neginf=max_range)
        depth_clean = torch.clamp(depth_clean, 0.0, max_range)
        # (N, H, W, 1) → (N, H*W) → min over pixels → (N,)
        self._min_depth_cache = depth_clean.flatten(1).min(dim=1).values

        # ── 2D VAE latent (depth → latent, matches depth-trained DCE) ────────
        depth_input = self.vae_encoder.preprocess_depth(depth)    # (N, 1, H, W) in [0, 1]
        latent_2d = self.vae_encoder.encode(depth_input)          # (N, 512)

        # ── 3D AE: one-pass batched build of OCC, SVS, and the reward features ─
        feats = self.autoencoder3D.build_local_features_batched()
        combined = feats["combined"]                                       # (N, 2, NZ, NY, NX)
        self._dist_to_obstacle_cache = feats["dist_to_obstacle"]           # (N,)
        self._Nt_cache               = feats["Nt"]                         # (N,)
        latent_3d = self.autoencoder3D.encode(combined)                    # (N, latent_dim_3d)

        # if self.common_step_counter % 30 == 0:
        #     # ── 3D AE: target (preprocessed) vs reconstructed (env 0) ──────────
        #     target = self.autoencoder3D.preprocess(combined[0:1])      # (1, 2, NZ, NY, NX)
        #     recon  = self.autoencoder3D.decode(latent_3d[0:1])         # (1, 2, NZ, NY, NX), sigmoid-ed
        #     target_np = target[0].detach().cpu().numpy()
        #     recon_np  = recon[0].detach().cpu().numpy()

        #     fig = plt.figure(figsize=(14, 8))
        #     fig.suptitle(f"Step {self.common_step_counter} — env 0 local maps", fontsize=12)

        #     for row, (data, label) in enumerate([(target_np, "Target"), (recon_np, "Recon")]):
        #         occ = data[0]
        #         svs = data[1]

        #         ax = fig.add_subplot(2, 2, row * 2 + 1, projection="3d")
        #         iz, iy, ix = np.where(occ > 0.5)
        #         if len(iz):
        #             ax.scatter(ix, iy, iz, s=20, c="red", alpha=0.4, marker="s")
        #         ax.set_xlim(0, occ.shape[2] - 1)
        #         ax.set_ylim(0, occ.shape[1] - 1)
        #         ax.set_zlim(0, occ.shape[0] - 1)
        #         ax.set_title(f"{label} OCC ({len(iz)} voxels)")
        #         ax.set_xlabel("body x"); ax.set_ylabel("body y"); ax.set_zlabel("body z")

        #         ax2 = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
        #         iz2, iy2, ix2 = np.where(svs > 0.15)
        #         if len(iz2):
        #             vals = svs[iz2, iy2, ix2]
        #             sc = ax2.scatter(ix2, iy2, iz2, s=20, c=vals,
        #                              cmap="viridis", alpha=0.5, marker="s",
        #                              vmin=0.0, vmax=1.0)
        #             fig.colorbar(sc, ax=ax2, shrink=0.5)
        #         ax2.set_xlim(0, svs.shape[2] - 1)
        #         ax2.set_ylim(0, svs.shape[1] - 1)
        #         ax2.set_zlim(0, svs.shape[0] - 1)
        #         ax2.set_title(f"{label} SVS")
        #         ax2.set_xlabel("body x"); ax2.set_ylabel("body y"); ax2.set_zlabel("body z")

        #     plt.tight_layout()
        #     plt.savefig("local_maps_check.png", dpi=120, bbox_inches="tight")
        #     plt.close()

        
        # if self.common_step_counter % 30 == 0:
        #     # ── 2D VAE: depth → VAE input → recon (env 0) in one figure ────────
        #     depth_np      = (depth[0, :, :, 0] / self.max_depth).detach().cpu().numpy()
        #     vae_input_np  = depth_input[0, 0].detach().cpu().numpy()
        #     recon_2d      = self.vae_encoder.decode(latent_2d[0:1])
        #     recon_np      = recon_2d[0, 0].detach().cpu().numpy()

        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #     fig.suptitle(f"Step {self.common_step_counter} — env 0 2D VAE", fontsize=12)

        #     titles = ["Depth (normalized)", "VAE input (depth)", "Reconstruction"]
        #     images = [depth_np, vae_input_np, recon_np]

        #     for ax, img, title in zip(axes, images, titles):
        #         im = ax.imshow(img, cmap="plasma", vmin=0, vmax=1)
        #         ax.set_title(title)
        #         ax.axis("off")

        #     fig.colorbar(im, ax=axes, shrink=0.7, fraction=0.02, pad=0.02)
        #     plt.savefig("vae2d_check.png", dpi=120, bbox_inches="tight")
        #     plt.close()

        obs = torch.cat(
            [
                self.rel_pos_b,                                      # 3
                self._prev_actions,                                  # 4
                self._robot.data.root_lin_vel_b,                     # 3
                self._robot.data.root_ang_vel_b,                     # 3
                self.distance_to_bounds_x.reshape(-1, 1).float(),    # 1
                self.distance_to_bounds_y.reshape(-1, 1).float(),    # 1
                latent_2d,                                           # 512
                latent_3d,                                           # 192
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:

        self._episode_sums["mean_dist_to_obstacle"] += self._dist_to_obstacle_cache * self.step_dt
        origins = self.scene.env_origins

        # ── Velocity regularization (exp reward, in [0, 1]) ───────────────────
        lin_vel_sum = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sum = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        lin_vel = torch.exp(-lin_vel_sum / self.cfg.lin_vel_tau)
        ang_vel = torch.exp(-ang_vel_sum / self.cfg.ang_vel_tau)

        # ── Goal-distance reward ──────────────────────────────────────────────
        d = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        prev_d = self._prev_dist_to_goal
        distance_to_goal_mapped = 1 - torch.tanh(self.final_distance_to_goal / 0.8)

        # r1 = torch.exp(-(d ** 2) / self.cfg.goal_nu1)           # narrow Gaussian, reward in [0, 1]
        # r2 = torch.exp(-(d ** 2) / self.cfg.goal_nu2)           # medium Gaussian, reward in [0, 1]
        # # raw progress (>0 approaching), only clamped to [-1, 1] as a safety guard.
        # r4 = torch.clamp(prev_d - d, -1.0, 1.0)

        # distance_to_goal_reward = (
        #     self.cfg.goal_lambda1 * r1
        #     + self.cfg.goal_lambda2 * r2
        #     + self.cfg.goal_progress_scale * r4
        # )

        inside = d.unsqueeze(1) < self._success_thresholds.unsqueeze(0)   # (N, K)
        newly_crossed = inside & (~self._success_claimed)                 # (N, K)
        success = newly_crossed.float().sum(dim=1)                        # (N,)
        self._success_claimed = self._success_claimed | inside

        # ── Action regularization (exp reward, in [0, 1]) ─────────────────────
        action_diff = self._actions - self._prev_actions
        action_reg_norm = torch.norm(action_diff, p=2, dim=-1)
        action_reg_diff = torch.exp(-action_reg_norm / self.cfg.action_reg_tau)

        is_alive_bounds    = torch.logical_and(self.distance_to_bounds_x, self.distance_to_bounds_y)
        is_alive_height    = torch.logical_and(
            self._robot.data.root_pos_w[:, 2] > self.z_min,
            self._robot.data.root_pos_w[:, 2] < self.z_max,
        )
        is_alive_collision = self._dist_to_obstacle_cache > self.cfg.collision_distance
        is_alive = is_alive_bounds & is_alive_height & is_alive_collision
        life = torch.where(is_alive, self.cfg.alive_reward_scale * self.step_dt, self.cfg.death_reward_scale)

        # ── Distance to obstacles (exp penalty, in [-1, 0]) ──────────────────
        dist_to_obs_reward = -torch.exp(-self._min_depth_cache / self.cfg.safety_radius)
        # ── Exploration (exp reward, in [0, 1]) ──────────────────────────────
        exploration_reward = torch.exp(-self.cfg.exploration_delta * self._Nt_cache)
        

        # se si è vicini alla posizione obiettivo, non incentivare più l'esplorazione, altrimenti il drone potrebbe essere tentato di allontanarsi per esplorare nuove celle.

        rewards = {
            "lin_vel":                  lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel":                  ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal":         distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_reg_diff":          action_reg_diff * self.cfg.rew_scale_action_reg * self.step_dt,
            "life":                     life,
            "distance_to_obstacles":    dist_to_obs_reward * self.cfg.distance_to_obstacles_reward_scale * self.step_dt,
            "exploration":              exploration_reward * self.cfg.exploration_reward_scale * self.step_dt,
            "success":                  success * self.cfg.goal_success_scale,

        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._prev_dist_to_goal = d.clone()

        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        out_of_bounds = (
            (self._robot.data.root_pos_w[:, 2] < self.z_min + 0.3) |
            (self._robot.data.root_pos_w[:, 2] > self.z_max - 0.3) |
            (~self.distance_to_bounds_x.bool()) |
            (~self.distance_to_bounds_y.bool())
        )
        collided = self._dist_to_obstacle_cache <= self.cfg.collision_distance

        died = out_of_bounds | collided
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        num_deaths           = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        num_timeouts         = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        num_deaths_collision = int((
            self.reset_terminated[env_ids] &
            (self._dist_to_obstacle_cache[env_ids] < self.cfg.collision_distance)
        ).sum().item())



        cells_visited_per_env = (
            self.autoencoder3D.global_visit_counts[env_ids] > 0
        ).flatten(1).sum(dim=1).float()
        self._episode_sums["cells_visited"][env_ids] = (
            cells_visited_per_env * self.max_episode_length_s
        )

        terminal_distance = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
            dim=1,
        )

        # ── Curriculum: count successes among ended episodes, advance level ───
        ended_mask = self.reset_terminated[env_ids] | self.reset_time_outs[env_ids]
        n_ended = int(ended_mask.sum().item())
        if n_ended > 0:
            success_mask = self.reset_time_outs[env_ids] & (terminal_distance < self.cfg.goal_radius)
            self._curr_episodes += n_ended
            self._curr_successes += int(success_mask.sum().item())
            if self._curr_episodes >= self.cfg.curriculum_window:
                success_rate = self._curr_successes / max(self._curr_episodes, 1)
                if success_rate >= self.cfg.curriculum_success_threshold:
                    self._curr_level = min(
                        self._curr_level + 1,
                        len(self._curriculum_schedule) - 1,
                    )
                self._curr_episodes = 0
                self._curr_successes = 0


        self._episode_sums["final_distance_to_goal"][env_ids] = terminal_distance * self.max_episode_length_s

        # ── Reset visit counts for terminated envs ────────────────────────────
        self.autoencoder3D.global_visit_counts[env_ids] = 0.0
        self.autoencoder3D.global_occ_map[env_ids] = 0

        self.extras["log"] = dict()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key in ["final_distance_to_goal", "mean_dist_to_obstacle", "cells_visited"]:
                extras["Episode_Info/" + key] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"].update(extras)
        self.extras["log"]["Curriculum/level"] = torch.tensor(float(self._curr_level + 1), device=self.device)
        self.extras["log"]["Curriculum/frac"] = torch.tensor(float(self._curriculum_schedule[self._curr_level][0]), device=self.device)
        n_reset = max(len(env_ids), 1)
        self.extras["log"]["Episode_Termination/died"] = torch.tensor(num_deaths / n_reset, device=self.device)
        self.extras["log"]["Episode_Termination/time_out"] = torch.tensor(num_timeouts / n_reset, device=self.device)
        self.extras["log"]["Episode_Termination/died_collision"] = torch.tensor(num_deaths_collision / n_reset, device=self.device)

        coll_mask = self.reset_terminated[env_ids] & (
            self._dist_to_obstacle_cache[env_ids] < self.cfg.collision_distance
        )
        if coll_mask.any():
            z_death   = self._robot.data.root_pos_w[env_ids][coll_mask, 2]
            len_death = self.episode_length_buf[env_ids][coll_mask].float()
            self.extras["log"]["Debug/coll_z_mean"]   = z_death.mean()
            self.extras["log"]["Debug/coll_steplen"]  = len_death.mean()


        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._prev_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self.alive_steps[env_ids] = 0
        self._reached_goal[env_ids] = False

        # gx, gy, gz = self._sample_in_curriculum_box(len(env_ids))
        # self._desired_pos_w[env_ids, 0] = gx
        # self._desired_pos_w[env_ids, 1] = gy
        # self._desired_pos_w[env_ids, 2] = gz

        # self._desired_pos_w[env_ids, 0] = torch.empty_like(
        #     self._desired_pos_w[env_ids, 0]
        # ).uniform_(self.x_min + 1.0, self.x_max - 1.0)
        # self._desired_pos_w[env_ids, 1] = torch.empty_like(
        #     self._desired_pos_w[env_ids, 1]
        # ).uniform_(self.y_min + 1.0, self.y_max - 1.0)
        # self._desired_pos_w[env_ids, 2] = torch.empty_like(
        #     self._desired_pos_w[env_ids, 2]
        # ).uniform_(self.z_min + 1.0, self.z_max - 1.0)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # dx, dy, dz = self._sample_in_curriculum_box(len(env_ids))
        # default_root_state[:, 0] = dx
        # default_root_state[:, 1] = dy
        # default_root_state[:, 2] = dz
        drone_pos, goal_pos = self._sample_drone_and_goal(len(env_ids))
        # # ── DEBUG: commanded spawn position ───────────────────────────────────
        # for k in range(len(env_ids)):
        #     ei = int(env_ids[k])
        #     eo = self.scene.env_origins[ei]
        #     print(f"[SPAWN] env {ei}: cmd_world=("
        #           f"{drone_pos[k,0]:.2f}, {drone_pos[k,1]:.2f}, {drone_pos[k,2]:.2f})  "
        #           f"goal=({goal_pos[k,0]:.2f}, {goal_pos[k,1]:.2f}, {goal_pos[k,2]:.2f})  "
        #           f"env_origin=({eo[0]:.2f}, {eo[1]:.2f}, {eo[2]:.2f})")

        
        self._desired_pos_w[env_ids] = goal_pos
        self._prev_dist_to_goal[env_ids] = torch.linalg.norm(goal_pos - drone_pos, dim=1)

        self._success_claimed[env_ids] = (
            self._prev_dist_to_goal[env_ids].unsqueeze(1) < self._success_thresholds.unsqueeze(0)
        )

        default_root_state[:, 0] = drone_pos[:, 0]
        default_root_state[:, 1] = drone_pos[:, 1]
        default_root_state[:, 2] = drone_pos[:, 2]

        # default_root_state[:, 0] = torch.empty_like(default_root_state[:, 0]).uniform_(self.x_min + 1.0, self.x_max - 1.0)
        # default_root_state[:, 1] = torch.empty_like(default_root_state[:, 1]).uniform_(self.y_min + 1.0, self.y_max - 1.0)
        # default_root_state[:, 2] = torch.empty_like(default_root_state[:, 2]).uniform_(self.z_min + 1.0, self.z_max - 1.0)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
