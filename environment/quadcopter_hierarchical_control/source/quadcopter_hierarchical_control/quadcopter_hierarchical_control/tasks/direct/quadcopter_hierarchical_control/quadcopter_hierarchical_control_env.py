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

from .quadcopter_hierarchical_control_env_cfg import QuadcopterHierarchicalControlEnvCfg
from isaaclab_assets import CRAZYFLIE_CFG  
from isaaclab.markers import CUBOID_MARKER_CFG  

import wandb

wandb.login()

# Project that the run is recorded to
project = "quadcopter_hierarchical_control"

class QuadcopterHierarchicalControlEnv(DirectRLEnv):
    cfg: QuadcopterHierarchicalControlEnvCfg

    def __init__(self, cfg: QuadcopterHierarchicalControlEnvCfg, render_mode: str | None = None, **kwargs):
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
                "action_reg_diff",
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
        self.arena_size = torch.ones(self.num_envs, device=self.device) * 4.0

        # Accumulatori per valutazione
        self.accum_deaths = 0.0
        self.accum_timeouts = 0.0
        self.accum_reward = 0.0
        self.accum_counter = 0

        # Reward massimo teorico per episodio (per normalizzare)
        max_theoretical_reward = self.cfg.alive_reward_scale * self.cfg.distance_to_goal_reward_scale * self.max_episode_length_s
        self.policy_network = self._load_policy_network()
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

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
        self.target_vel_cmd[:,:3] = self._actions[:, :3]
        self.target_yaw_cmd = self._actions[:, 3]

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
        self._prev_actions = self._actions.clone()  
        obs = torch.cat(
            [
                self._prev_actions,
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
        # 3. Action Regularization

        action_diff = self._actions - self._prev_actions  # (num_envs, 4)

        action_reg_diff = torch.norm(action_diff, p=2, dim=-1)  # (num_envs,)
        action_reg_diff = 1 - torch.tanh(action_reg_diff / 0.8) # (num_envs,)
        is_alive = torch.logical_and(self.distance_to_bounds_x >= 0.0, self.distance_to_bounds_y >= 0.0)
        life = torch.where(is_alive, 
                        self.cfg.alive_reward_scale, 
                        self.cfg.death_reward_scale)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_reg_diff": action_reg_diff * self.cfg.rew_scale_action_reg * self.step_dt,
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
        
        self._prev_actions = 0.0
        self._actions[env_ids] = 0.0
        #self._prev_actions[env_ids] = 0.0
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = 1.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


        
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

