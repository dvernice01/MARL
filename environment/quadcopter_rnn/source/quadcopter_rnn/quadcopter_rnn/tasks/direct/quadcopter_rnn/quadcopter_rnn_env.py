# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import torch
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
from isaaclab.utils.math import subtract_frame_transforms
import isaaclab.utils.math as math_utils

from .quadcopter_rnn_env_cfg import QuadcopterRnnEnvCfg
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.markers import CUBOID_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, RayCaster
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.mplot3d import Axes3D


wandb.login()

project = "quadcopter_rnn"

# ── Paths ──────────────────────────────────────────────────────────────────────
OCCUPANCY_MAP_PATH = "/workspace/environment/quadcopter_rnn/warehouse_3d/occupancy_3d.npy"
OCCUPANCY_META_PATH = "/workspace/environment/quadcopter_rnn/warehouse_3d/occupancy_3d_meta.npy"
LOCAL_MAPS_SAVE_DIR = "/workspace/environment/quadcopter_rnn/outputs/local_maps"
LOCAL_MAP_SAVE_EVERY = 100   # steps between saves; set to 0 to disable
LOCAL_N = 32                 # local cube size: n x n x n voxels


def visualize_occupancy_3d(occ_map):
    ix, iy, iz = torch.where(occ_map > 0)
    ix = ix.detach().cpu()
    iy = iy.detach().cpu()
    iz = iz.detach().cpu()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ix.numpy(), iy.numpy(), iz.numpy(), s=1, c='red', alpha=0.3)
    plt.show()
    plt.savefig('occupancy_map.png')
    plt.close()


def hits_to_occupancy_map(ray_hits_w, grid_size=0.05, map_dims=(200, 200, 100), origin=None):
    pts = ray_hits_w.reshape(-1, 3)
    valid = torch.isfinite(pts).all(dim=-1)
    pts = pts[valid]
    if origin is None:
        origin = torch.zeros(3, device=pts.device)
    else:
        origin = torch.tensor(origin, device=pts.device, dtype=pts.dtype)
    cx = map_dims[0] // 2
    cy = map_dims[1] // 2
    cz = map_dims[2] // 2
    ix = ((pts[:, 0] - origin[0]) / grid_size + cx).long()
    iy = ((pts[:, 1] - origin[1]) / grid_size + cy).long()
    iz = ((pts[:, 2] - origin[2]) / grid_size + cz).long()
    mask = (
        (ix >= 0) & (ix < map_dims[0]) &
        (iy >= 0) & (iy < map_dims[1]) &
        (iz >= 0) & (iz < map_dims[2])
    )
    ix, iy, iz = ix[mask], iy[mask], iz[mask]
    occ_map = torch.zeros(map_dims, dtype=torch.float32, device=pts.device)
    occ_map[ix, iy, iz] = 1.0
    return occ_map


class QuadcopterRnnEnv(DirectRLEnv):
    cfg: QuadcopterRnnEnvCfg

    def __init__(self, cfg: QuadcopterRnnEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.low_level_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_yaw_cmd = torch.zeros(self.num_envs, 1, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.rel_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

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

        self.set_debug_vis(self.cfg.debug_vis)

        self.distance_to_bounds_x = torch.zeros(self.num_envs, device=self.device)
        self.distance_to_bounds_y = torch.zeros(self.num_envs, device=self.device)
        self.final_distance_to_goal_b = torch.zeros(self.num_envs, device=self.device)

        self.curriculum_level = 0
        self.arena_size = torch.ones(self.num_envs, device=self.device) * 4.0

        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0

        self.policy_network = self._load_policy_network()
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # ── SECTION 1: Load global occupancy map ──────────────────────────────
        occ_meta = np.load(OCCUPANCY_META_PATH, allow_pickle=True).item()
        self.global_occ_map = torch.tensor(
            np.load(OCCUPANCY_MAP_PATH),
            dtype=torch.float32, device=self.device
        )  # shape: (n_z, n_y, n_x)

        self.occ_grid_size = float(occ_meta["cell_size"])
        self.occ_origin = torch.tensor(
            [occ_meta["x_min"], occ_meta["y_min"], occ_meta["z_min"]],
            dtype=torch.float32, device=self.device
        )  # world coords of voxel [0, 0, 0]
        self.occ_map_dims = self.global_occ_map.shape  # (n_z, n_y, n_x)

        # ── Local map parameters ──────────────────────────────────────────────
        self.local_n = LOCAL_N
        self.local_r_vox = LOCAL_N // 2  # half-extent in voxels

        # ── SECTION 2: Visit count grid — (num_envs, n_z, n_y, n_x) ─────────
        self.global_visit_counts = torch.zeros(
            (self.num_envs, *self.occ_map_dims),
            dtype=torch.float32, device=self.device
        )

        os.makedirs(LOCAL_MAPS_SAVE_DIR, exist_ok=True)
        print(f"[LocalMap] Occupancy map loaded: shape={self.occ_map_dims}, cell={self.occ_grid_size}m")

    # ── SECTION 2: Update visit counts (call every step) ──────────────────────
    def _update_visit_counts(self, env_ids: torch.Tensor | None = None):
        """Increment the visit count voxel at each drone's current position."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        pos_w = self._robot.data.root_pos_w[env_ids]  # (E, 3)

        iz = ((pos_w[:, 2] - self.occ_origin[2]) / self.occ_grid_size).long()
        iy = ((pos_w[:, 1] - self.occ_origin[1]) / self.occ_grid_size).long()
        ix = ((pos_w[:, 0] - self.occ_origin[0]) / self.occ_grid_size).long()

        NZ, NY, NX = self.occ_map_dims
        valid = (iz >= 0) & (iz < NZ) & (iy >= 0) & (iy < NY) & (ix >= 0) & (ix < NX)

        for i, e in enumerate(env_ids):
            if valid[i]:
                self.global_visit_counts[e, iz[i], iy[i], ix[i]] += 1.0

    # ── SECTION 2: Build local SVS map ────────────────────────────────────────
    def _build_local_svs_map(self, env_id: int) -> torch.Tensor:
        """
        Crop local n x n x n visit count region around the drone,
        normalize and apply Shannon entropy per cell.
        Returns: (n_z, n_y, n_x) SVS map.
        """
        pos_w = self._robot.data.root_pos_w[env_id]
        cz = int((pos_w[2] - self.occ_origin[2]) / self.occ_grid_size)
        cy = int((pos_w[1] - self.occ_origin[1]) / self.occ_grid_size)
        cx = int((pos_w[0] - self.occ_origin[0]) / self.occ_grid_size)

        h = self.local_r_vox
        n = self.local_n
        NZ, NY, NX = self.occ_map_dims

        svs = torch.zeros((n, n, n), dtype=torch.float32, device=self.device)

        z0, z1 = cz - h, cz + h
        y0, y1 = cy - h, cy + h
        x0, x1 = cx - h, cx + h

        sz0, sz1 = max(z0, 0), min(z1, NZ)
        sy0, sy1 = max(y0, 0), min(y1, NY)
        sx0, sx1 = max(x0, 0), min(x1, NX)

        dz0, dz1 = sz0 - z0, sz1 - z0
        dy0, dy1 = sy0 - y0, sy1 - y0
        dx0, dx1 = sx0 - x0, sx1 - x0

        if sz0 < sz1 and sy0 < sy1 and sx0 < sx1:
            counts = self.global_visit_counts[env_id, sz0:sz1, sy0:sy1, sx0:sx1]
            Nt = counts.sum()
            if Nt > 0:
                p = counts / Nt
                entropy = torch.where(p > 0, -p * torch.log(p), torch.zeros_like(p))
                svs[dz0:dz1, dy0:dy1, dx0:dx1] = entropy

        return svs  # (n, n, n)

    # ── SECTION 3: Sample local occupancy map from global ─────────────────────
    def _build_local_occ_map(self, env_id: int) -> torch.Tensor:
        """
        Crop n x n x n local occupancy map centered on the drone.
        OCCUPIED (==2) → 1.0, FREE/UNKNOWN → 0.0.
        Returns: (n_z, n_y, n_x) binary occupancy map.
        """
        pos_w = self._robot.data.root_pos_w[env_id]
        cz = int((pos_w[2] - self.occ_origin[2]) / self.occ_grid_size)
        cy = int((pos_w[1] - self.occ_origin[1]) / self.occ_grid_size)
        cx = int((pos_w[0] - self.occ_origin[0]) / self.occ_grid_size)

        h = self.local_r_vox
        n = self.local_n
        NZ, NY, NX = self.occ_map_dims

        local_occ = torch.zeros((n, n, n), dtype=torch.float32, device=self.device)

        z0, z1 = cz - h, cz + h
        y0, y1 = cy - h, cy + h
        x0, x1 = cx - h, cx + h

        sz0, sz1 = max(z0, 0), min(z1, NZ)
        sy0, sy1 = max(y0, 0), min(y1, NY)
        sx0, sx1 = max(x0, 0), min(x1, NX)

        dz0, dz1 = sz0 - z0, sz1 - z0
        dy0, dy1 = sy0 - y0, sy1 - y0
        dx0, dx1 = sx0 - x0, sx1 - x0

        if sz0 < sz1 and sy0 < sy1 and sx0 < sx1:
            raw = self.global_occ_map[sz0:sz1, sy0:sy1, sx0:sx1]
            local_occ[dz0:dz1, dy0:dy1, dx0:dx1] = (raw == 2).float()

        return local_occ  # (n, n, n)

    # ── SECTION 4: Stack the two maps as separate channels ────────────────────
    def _build_local_combined_map(self, env_id: int) -> torch.Tensor:
        """
        Returns (2, n, n, n):
          channel 0 = local binary occupancy
          channel 1 = local SVS (Shannon entropy of visit distribution)
        Stacking (not summing) preserves the semantic independence of the two maps.
        """
        local_occ = self._build_local_occ_map(env_id)
        local_svs = self._build_local_svs_map(env_id)
        return torch.stack([local_occ, local_svs], dim=0)  # (2, n, n, n)

    # ── SECTION 5: Save combined maps ─────────────────────────────────────────
    def _save_local_maps(self):
        """Save combined (2, n, n, n) maps for all envs to disk."""
        step = self.common_step_counter
        for env_id in range(self.num_envs):
            combined = self._build_local_combined_map(env_id)  # (2, n, n, n)
            path = os.path.join(LOCAL_MAPS_SAVE_DIR, f"env{env_id}_step{step}.npy")
            np.save(path, combined.cpu().numpy())

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
        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
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
        self._update_visit_counts()

        # ── Save combined maps periodically ───────────────────────────────────
        if LOCAL_MAP_SAVE_EVERY > 0 and self.common_step_counter % LOCAL_MAP_SAVE_EVERY == 0:
            self._save_local_maps()

        self.rel_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w
        )
        self.final_distance_to_goal = torch.linalg.norm(self.rel_pos_b, dim=1)

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
        relative_distance_to_bounds_x = torch.abs((origins[:, 0]) - self._robot.data.root_pos_w[:, 0])
        relative_distance_to_bounds_y = torch.abs((origins[:, 1]) - self._robot.data.root_pos_w[:, 1])
        self.distance_to_bounds_x = (square_side / 2) - relative_distance_to_bounds_x
        self.distance_to_bounds_y = (square_side / 2) - relative_distance_to_bounds_y

        action_diff = self._actions - self._prev_actions
        action_reg_diff = torch.norm(action_diff, p=2, dim=-1)
        action_reg_diff = 1 - torch.tanh(action_reg_diff / 0.8)

        is_alive = torch.logical_and(self.distance_to_bounds_x >= 0.0, self.distance_to_bounds_y >= 0.0)
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
        died = (self._robot.data.root_pos_w[:, 2] < 0.1) | \
               (self._robot.data.root_pos_w[:, 2] > 2.0) | \
               (self.distance_to_bounds_x < 0.0) | \
               (self.distance_to_bounds_y < 0.0)
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

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = 1.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Reset visit counts for terminated envs ────────────────────────────
        self.global_visit_counts[env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass