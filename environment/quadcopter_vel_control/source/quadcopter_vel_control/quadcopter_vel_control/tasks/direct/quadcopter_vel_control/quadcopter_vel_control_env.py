# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform
from .quadcopter_vel_control_env_cfg import QuadcopterVelControlEnvCfg
import wandb



class QuadcopterVelControlEnv(DirectRLEnv):
    cfg: QuadcopterVelControlEnvCfg

    def __init__(self, cfg: QuadcopterVelControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Reference commands
        self._lin_vel_ref = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_vel_ref = torch.zeros(self.num_envs, 1, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_tracking",
                "ang_vel_tracking",
                "action_reg_thrust",
                "action_reg_moments",
                "alive",
                "tilt_penalty",
                "lin_vel_mae",
                "ang_vel_mae",
            ]
        }
        
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Episode counter for canonical sampling
        self._episode_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # Thrust (Z-axis force)
        # action[0] is mapped from [-1, 1] to [0, 1]
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # Moments (Roll, Pitch, Yaw)
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # Dynamic Command Resampling
        # Resample commands with a small probability every step to simulate dynamic targets
        # Probability per step = dt / interval
        if self.cfg.resample_command_interval_s > 0.0:
            prob = self.step_dt / self.cfg.resample_command_interval_s
            resample_mask = torch.rand(self.num_envs, device=self.device) < prob
            # Avoid resampling if just reset? No easy way to track "just reset" here without extra buffer.
            # But probability is low, so it's fine.
            env_ids_to_resample = resample_mask.nonzero(as_tuple=False).squeeze(-1) # TROVA GLI ELEMENTI NON NULLI
            if len(env_ids_to_resample) > 0:
                self._sample_commands(env_ids_to_resample)

        # Self state
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        proj_grav_b = self._robot.data.projected_gravity_b
        
        # References
        # lin_vel_ref is already in body frame (as we sample it directly or assume it's given in body frame)
        # Wait, usually references are given in world frame and transformed, but for a velocity controller 
        # it's common to give commands in body frame (e.g. "move forward").
        # Let's assume the reference IS in body frame for simplicity of the low-level controller.
        
        obs = torch.cat(
            [
                lin_vel_b,          # 3
                ang_vel_b,          # 3
                proj_grav_b,        # 3
                self._lin_vel_ref,  # 3
                self._yaw_vel_ref,  # 1
            ],
            dim=-1,
        ) # Total 13
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # 1. Linear Velocity Tracking
        # Huber Loss between current body velocity and reference body velocity
        lin_vel_loss = F.huber_loss(self._robot.data.root_lin_vel_b, self._lin_vel_ref, reduction='none', delta=self.cfg.huber_delta)
        lin_vel_error = torch.sum(lin_vel_loss, dim=1)
        lin_vel_tracking = lin_vel_error * self.cfg.rew_scale_lin_vel_tracking * self.step_dt
        
        # 2. Angular Velocity Tracking (Yaw only)
        # Huber Loss between current yaw rate and reference yaw rate
        ang_vel_loss = F.huber_loss(self._robot.data.root_ang_vel_b[:, 2], self._yaw_vel_ref.squeeze(-1), reduction='none', delta=self.cfg.huber_delta)
        ang_vel_error = ang_vel_loss
        ang_vel_tracking = ang_vel_error * self.cfg.rew_scale_ang_vel_tracking * self.step_dt
        
        # 3. Action Regularization
        # Separate thrust (action[0]) and moments (action[1:])
        action_reg_thrust = torch.square(self._actions[:, 0]) * self.cfg.rew_scale_action_reg * self.step_dt
        action_reg_moments = torch.sum(torch.square(self._actions[:, 1:]), dim=1) * self.cfg.rew_scale_action_reg * self.step_dt
        
        # 4. Alive Reward
        # We only give this if the drone is not terminated (handled in _get_dones logic implicitly via reset, 
        # but here we just give it every step. If it dies, it resets, so it loses potential future rewards).
        alive = self.cfg.rew_scale_alive * self.step_dt
        
        # 5. Tilt Penalty (Termination)
        # Check tilt condition (same as in _get_dones)
        is_tilted = self._robot.data.projected_gravity_b[:, 2] < -0.2 # Wait, logic was > -0.2 in _get_dones?
        # Let's verify the logic. 
        # Upright: (0, 0, -1). Tilted 90 deg: (0, 0, 0). Tilted 180 deg: (0, 0, 1).
        # So Z component goes from -1 to 1.
        # If we want max tilt 80 deg (approx). cos(80) = 0.17.
        # So Z component should be < -0.17 (closer to -1).
        # If Z > -0.17 (closer to 0 or positive), it is tilted too much.
        # So is_tilted = proj_grav_b[:, 2] > -0.2 is correct for "bad state".
        
        is_tilted = self._robot.data.projected_gravity_b[:, 2] > -0.2
        tilt_penalty = is_tilted.float() * self.cfg.rew_scale_tilt_penalty
        
        rewards = {
            "lin_vel_tracking": lin_vel_tracking,
            "ang_vel_tracking": ang_vel_tracking,
            "action_reg_thrust": action_reg_thrust,
            "action_reg_moments": action_reg_moments,
            "alive": torch.full_like(lin_vel_tracking, alive),
            "tilt_penalty": tilt_penalty,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        # Log MAE (Accumulate error * dt, so average over time gives Mean Absolute Error)
        lin_vel_error_norm = torch.norm(self._robot.data.root_lin_vel_b - self._lin_vel_ref, dim=1)
        ang_vel_error_abs = torch.abs(self._robot.data.root_ang_vel_b[:, 2] - self._yaw_vel_ref.squeeze(-1))
        
        self._episode_sums["lin_vel_mae"] += lin_vel_error_norm * self.step_dt
        self._episode_sums["ang_vel_mae"] += ang_vel_error_abs * self.step_dt
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Termination conditions
        # 1. Tilt (Projected gravity on Z axis < threshold)
        # proj_grav_b[2] = cos(theta). If theta > 80 deg (1.4 rad), cos(theta) < 0.17
        # Let's use the config value.
        # We can check projected gravity Z component. 
        # If upright, it is -1 (gravity points down in body frame). Wait, Isaac Lab convention?
        # projected_gravity_b is R^T * [0, 0, -1].
        # If upright (R=I), it is [0, 0, -1].
        # So we check if z component is > -0.17 (meaning it's close to 0, i.e., 90 deg tilt) or > some threshold.
        # Actually, let's look at how it's usually done. 
        # If we want max tilt 80 deg.
        # Let's just use the Z position of projected gravity.
        # If it is too small (close to 0), it means gravity is perpendicular to Z axis (90 deg tilt).
        # If it is -1, it is upright.
        # So we want proj_grav_b[2] < -cos(max_tilt).
        # cos(1.4) ~= 0.17. So we want proj_grav_b[2] < -0.17.
        # Wait, gravity vector is (0, 0, -1) in world.
        # In body frame: R^T * (0, 0, -1).
        # If R is identity, result is (0, 0, -1).
        # So valid range is [-1, -0.17].
        # If it becomes > -0.17 (e.g. 0 or positive), it's tilted too much or upside down.
        
        # Simpler check: just check if z-axis of body is pointing up.
        # But we have projected gravity.
        # Let's assume we terminate if proj_grav_b[2] > -0.2 (approx).
        
        is_tilted = self._robot.data.projected_gravity_b[:, 2] > -0.2
        
        # 2. Ground collision (REMOVED)
        # is_ground_hit = self._robot.data.root_pos_w[:, 2] < self.cfg.min_z_pos
        
        died = is_tilted
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            
            # Separate actual reward components from diagnostic metrics
            if key in ["lin_vel_mae", "ang_vel_mae"]:
                extras["Episode_Info/" + key] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            
            self._episode_sums[key][env_ids] = 0.0
            
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        # Sample new commands
        # Increment episode counts
        self._episode_counts[env_ids] += 1
        
        self._sample_commands(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Randomize initial Z slightly to avoid ground collision immediately
        default_root_state[:, 2] += sample_uniform(self.cfg.initial_z_min, self.cfg.initial_z_max, (len(env_ids),), device=self.device)
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_commands(self, env_ids: torch.Tensor):
        # Determine canonical vs normal envs
        is_canonical = (self._episode_counts[env_ids] % self.cfg.canonical_interval == 0)
        canonical_ids = env_ids[is_canonical]
        normal_ids = env_ids[~is_canonical]
        
        # --- Normal Sampling ---
        if len(normal_ids) > 0:
            self._lin_vel_ref[normal_ids, 0] = sample_uniform(
                self.cfg.lin_vel_ref_range[0], self.cfg.lin_vel_ref_range[1], (len(normal_ids),), device=self.device
            )
            self._lin_vel_ref[normal_ids, 1] = sample_uniform(
                self.cfg.lin_vel_ref_range[0], self.cfg.lin_vel_ref_range[1], (len(normal_ids),), device=self.device
            )
            self._lin_vel_ref[normal_ids, 2] = sample_uniform(
                self.cfg.lin_vel_ref_range[0], self.cfg.lin_vel_ref_range[1], (len(normal_ids),), device=self.device
            )
            self._yaw_vel_ref[normal_ids, 0] = sample_uniform(
                self.cfg.yaw_vel_ref_range[0], self.cfg.yaw_vel_ref_range[1], (len(normal_ids),), device=self.device
            )

        # --- Canonical Sampling ---
        if len(canonical_ids) > 0:
            # Modes: 0=Hover, 1=+X, 2=-X, 3=+Y, 4=-Y, 5=+Z, 6=-Z, 7=+Yaw, 8=-Yaw
            modes = torch.randint(0, 9, (len(canonical_ids),), device=self.device)
            
            # Initialize to zero
            self._lin_vel_ref[canonical_ids] = 0.0
            self._yaw_vel_ref[canonical_ids] = 0.0
            
            # Helper for random magnitude in range [0.2, 1.0] (approx)
            # Assuming range is [-1, 1], we want [0.2, 1.0] for positive and [-1.0, -0.2] for negative
            # We can use the config range max.
            max_lin = self.cfg.lin_vel_ref_range[1]
            max_yaw = self.cfg.yaw_vel_ref_range[1]
            min_mag = 0.2 # Minimum magnitude to be distinct from hover
            
            # +X
            mask = (modes == 1)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 0] = sample_uniform(min_mag, max_lin, (mask.sum(),), device=self.device)
            # -X
            mask = (modes == 2)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 0] = sample_uniform(-max_lin, -min_mag, (mask.sum(),), device=self.device)
            # +Y
            mask = (modes == 3)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 1] = sample_uniform(min_mag, max_lin, (mask.sum(),), device=self.device)
            # -Y
            mask = (modes == 4)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 1] = sample_uniform(-max_lin, -min_mag, (mask.sum(),), device=self.device)
            # +Z
            mask = (modes == 5)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 2] = sample_uniform(min_mag, max_lin, (mask.sum(),), device=self.device)
            # -Z
            mask = (modes == 6)
            if mask.any():
                self._lin_vel_ref[canonical_ids[mask], 2] = sample_uniform(-max_lin, -min_mag, (mask.sum(),), device=self.device)
            # +Yaw
            mask = (modes == 7)
            if mask.any():
                self._yaw_vel_ref[canonical_ids[mask], 0] = sample_uniform(min_mag, max_yaw, (mask.sum(),), device=self.device)
            # -Yaw
            mask = (modes == 8)
            if mask.any():
                self._yaw_vel_ref[canonical_ids[mask], 0] = sample_uniform(-max_yaw, -min_mag, (mask.sum(),), device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # No debug visualization for now
        pass

    def _debug_vis_callback(self, event):
        pass
