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

from .quadcopter_env_cfg import QuadcopterEnvCfg
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

import wandb

wandb.login()

# Project that the run is recorded to
project = "quadcopter"

# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- INIZIALIZZAZIONE W&B ---
        # Verifichiamo se un run è già attivo per evitare duplicati
        if wandb.run is None:
            wandb.init(
                project="quadcopter",
                config={
                    "arena_shrink_rate": 0.002,
                    "arena_min": 0.5,
                    "lr": 0.01, 
                }
            )

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "life",
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

        # INIT PER CURRICULUM LEARNING
        self.curriculum_level = 0
        self.arena_size = torch.ones(self.num_envs, device=self.device) * 2.0

        # Accumulatori per valutazione
        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0

        # Reward massimo teorico per episodio (per normalizzare)
        max_theoretical_reward = self.cfg.alive_reward_scale * self.cfg.distance_to_goal_reward_scale * self.max_episode_length_s


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        #root_pos_relative = self._robot.data.root_pos_w - self.scene.env_origins
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._robot.data.root_pos_w,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        origins = self.scene.env_origins
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        
        square_side = self.arena_size  
        relative_distance_to_bounds_x = torch.abs((origins[:, 0]) - self._robot.data.root_pos_w[: ,0])
        relative_distance_to_bounds_y = torch.abs((origins[:, 1]) - self._robot.data.root_pos_w[: ,1])
        self.distance_to_bounds_x = square_side - relative_distance_to_bounds_x
        self.distance_to_bounds_y = square_side - relative_distance_to_bounds_y

        # Sostituisci il blocco if/else con questo:
        is_alive = torch.logical_and(self.distance_to_bounds_x >= 0.0, self.distance_to_bounds_y >= 0.0)

        # Se is_alive è True usa alive_reward_scale, altrimenti usa death_reward_scale
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
                (self._robot.data.root_pos_w[:, 2] > 2.0) | \
                (self.distance_to_bounds_x < 0.0) | \
                (self.distance_to_bounds_y < 0.0)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        num_deaths = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        num_timeouts = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        num_resets = len(env_ids)

        wandb.log({
            "Episode_Termination/died": num_deaths,
            "Episode_Termination/time_out": num_timeouts,
            "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
        })

        # CURRICULUM LEARNING
    # --- ACCUMULA STATISTICHE ---
        self.accum_deaths += num_deaths / num_resets          # death rate parziale
        self.accum_timeouts += num_timeouts / num_resets      # timeout rate parziale
        episode_reward = sum(
            torch.mean(self._episode_sums[key][env_ids]).item()
            for key in self._episode_sums.keys()
        ) / self.max_episode_length_s
        self.accum_reward += episode_reward
        self.accum_counter += 1

        self.extras["log"] = dict()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # azzerato DOPO aver letto
        self.extras["log"].update(extras)

        # --- VALUTAZIONE OGNI max_episode_length TIMESTEP ---
        if self.common_step_counter % self.max_episode_length == 0 and self.accum_counter > 0:
            
            death_rate = self.accum_deaths / self.accum_counter
            timeout_rate = self.accum_timeouts / self.accum_counter
            reward_media = self.accum_reward / self.accum_counter
            max_theoretical_reward = self.cfg.alive_reward_scale * self.max_episode_length_s
            reward_ratio = reward_media / max_theoretical_reward if max_theoretical_reward > 0 else 0

            # Log curriculum su TensorBoard
            wandb.log({
                "Curriculum/level": float(self.curriculum_level),
                "Curriculum/death_rate": death_rate,
                "Curriculum/timeout_rate": timeout_rate,
                "Curriculum/reward_ratio": reward_ratio,
                "Curriculum/arena_size_mean": self.arena_size.mean().item(),
            })

            # Condizione avanzamento livello
            if death_rate < 0.1 and timeout_rate > 0.9 and reward_ratio > 0.6:
                self.curriculum_level = min(self.curriculum_level + 1, 3)
                print(f"[Curriculum] Avanzamento a livello {self.curriculum_level}!")

            # Reset accumulatori
            self.accum_deaths = 0.0
            self.accum_timeouts = 0.0
            self.accum_reward = 0.0
            self.accum_counter = 0

        if self.curriculum_level == 0:
            self.arena_size[env_ids] = 2.0
            spawn_offset = torch.zeros(len(env_ids), 2, device=self.device)
        
        elif self.curriculum_level == 1:
            # Arena casuale [1.0, 3.0], robot al centro
            self.arena_size[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
            spawn_offset = torch.zeros(len(env_ids), 2, device=self.device)

        elif self.curriculum_level == 2:
            # Arena casuale [0.5, 3.5], robot casuale [-0.5, +0.5]
            self.arena_size[env_ids] = torch.rand(len(env_ids), device=self.device) * 3.0 + 0.5
            spawn_offset = (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * 0.5

        else:  # livello 3
            # Arena casuale [1.0, 3.0], robot casuale [-1.0, +1.0]
            self.arena_size[env_ids] = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
            spawn_offset = (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * 1.0

        # --- SPAWN TARGET ---
        target_offsets = (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * self.arena_size[env_ids].unsqueeze(1)
        self._desired_pos_w[env_ids, :2] = target_offsets + self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)


        #RESET DEGLI ENVIRONMENTS
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        #spawn_offsets = (torch.rand(len(env_ids), 2, device=self.device) * 2 - 1) * self.arena_size[env_ids].unsqueeze(1)
        default_root_state[:, :2] += spawn_offset
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)