"""
RL-policy-based 3D local map data collection.

Uses a pre-trained position controller to navigate the drone in the
quadcopter_rnn warehouse environment. Spawns within a restricted region
and only saves maps that contain occupancy.

Usage (from environment/quadcopter_rnn/):
  python scripts/skrl/collect_dataset.py \
    --task=Template-Quadcopter-Rnn-Direct-v0 \
    --checkpoint=/workspace/environment/quadcopter_hierarchical_control/projects/... \
    --num_envs=4 \
    --target_samples=1000 \
    --seed=-1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect 3D local map dataset via exploration.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Template-Quadcopter-Rnn-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to pre-trained position controller checkpoint")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--target_samples", type=int, default=1000,
                    help="Number of new samples to collect")
parser.add_argument("--output", type=str, default="outputs/dataset_3d_collection",
                    help="Output directory for .npy samples")
parser.add_argument("--goal_reached_dist", type=float, default=0.5,
                    help="Distance threshold to consider goal reached (m)")
parser.add_argument("--goal_timeout_steps", type=int, default=300,
                    help="Max steps before giving up on a goal")
parser.add_argument("--min_occ_voxels", type=int, default=10,
                    help="Min occupied voxels in local map to save sample")
parser.add_argument("--ml_framework", type=str, default="torch")
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--real-time", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import numpy as np
import skrl
import torch
import torch.nn as nn
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}.")
    exit()

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadcopter_rnn.tasks  # noqa: F401 — triggers gym.register

# Disable automatic map saving — we save manually
from quadcopter_rnn.tasks.direct.quadcopter_rnn import quadcopter_rnn_env as rnn_mod
rnn_mod.LOCAL_MAP_SAVE_EVERY = 0

# Restricted bounds (warehouse region with obstacles)
X_MIN, X_MAX = -28.0, 8.0
Y_MIN, Y_MAX = 0.0, 33.42
Z_MIN, Z_MAX = 0.0, 6.0
MARGIN = 0.5


# ── Custom models matching the checkpoint architecture ────────────────────────

class VelocityControllerPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256,
                 clip_actions=False, clip_log_std=True, min_log_std=-20,
                 max_log_std=2, initial_log_std=0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std, dtype=torch.float32)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        return self.net(x), self.log_std_parameter, {}


class VelocityControllerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        return self.net(x), {}


# ── Goal selection ────────────────────────────────────────────────────────────

def pick_goal_near_obstacle(occupied_bounded, device):
    """Pick a goal near a random obstacle voxel, offset 1-2.5m into free space.
    Returns goal in LOCAL warehouse coords (no env origin)."""
    idx = torch.randint(0, occupied_bounded.shape[0], (1,), device=device).item()
    base = occupied_bounded[idx].clone()

    direction = torch.randn(3, device=device)
    direction = direction / direction.norm()
    dist = random.uniform(1.0, 2.5)
    goal = base + direction * dist

    goal[0] = torch.clamp(goal[0], X_MIN + MARGIN, X_MAX - MARGIN)
    goal[1] = torch.clamp(goal[1], Y_MIN + MARGIN, Y_MAX - MARGIN)
    goal[2] = torch.clamp(goal[2], Z_MIN + MARGIN, Z_MAX - MARGIN)

    return goal


# ── Main ──────────────────────────────────────────────────────────────────────

agent_cfg_entry_point = "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         experiment_cfg: dict):

    # ── Seed ──────────────────────────────────────────────────────────────
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    print(f"[INFO] Seed: {args_cli.seed}")

    # ── Env config overrides ──────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.episode_length_s = 1000.0  # long episodes for exploration

    # ── Create env ────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

    # Override bounds to restrict spawn zone and death zone
    raw_env.z_max = Z_MAX

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env.device
    num_envs = args_cli.num_envs

    # ── Load pre-trained position controller with scaler ──────────────────
    models = {
        "policy": VelocityControllerPolicy(
            env.observation_space, env.action_space, device, hidden_size=512
        ),
        "value": VelocityControllerValue(
            env.observation_space, env.action_space, device, hidden_size=512
        ),
    }

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    agent_cfg["experiment"]["write_interval"] = 0
    agent_cfg["experiment"]["checkpoint_interval"] = 0

    agent = PPO(
        models=models,
        memory=RandomMemory(memory_size=64, num_envs=env.num_envs, device=device),
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")

    # ── Filter occupied voxels to restricted bounds ───────────────────────
    occupied = raw_env.occupied_pos_w  # (N, 3) local warehouse coords
    mask = (
        (occupied[:, 0] >= X_MIN) & (occupied[:, 0] <= X_MAX) &
        (occupied[:, 1] >= Y_MIN) & (occupied[:, 1] <= Y_MAX) &
        (occupied[:, 2] >= Z_MIN) & (occupied[:, 2] <= Z_MAX)
    )
    occupied_bounded = occupied[mask]
    print(f"[INFO] {occupied_bounded.shape[0]} occupied voxels within bounds")

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = args_cli.output
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".npy")])
    print(f"[INFO] Output: {output_dir} ({existing} existing samples)")

    # ── Exploration loop ──────────────────────────────────────────────────
    target = args_cli.target_samples
    goal_dist_thresh = args_cli.goal_reached_dist
    goal_timeout = args_cli.goal_timeout_steps
    min_occ = args_cli.min_occ_voxels

    obs, _ = env.reset()

    # Set initial goals near obstacles
    for eid in range(num_envs):
        goal = pick_goal_near_obstacle(occupied_bounded, device)
        raw_env._desired_pos_w[eid] = goal

    steps_on_goal = torch.zeros(num_envs, device=device)
    total_saved = existing
    total_reached = 0
    total_timeout = 0
    total_skipped_empty = 0
    step_count = 0

    print(f"[INFO] Collecting {target} new samples (min {min_occ} occ voxels per sample)...")

    while (total_saved - existing) < target and simulation_app.is_running():
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

        steps_on_goal += 1
        step_count += 1

        for eid in range(num_envs):
            is_term = bool(terminated[eid]) if terminated is not None else False
            is_trunc = bool(truncated[eid]) if truncated is not None else False

            if is_term or is_trunc:
                goal = pick_goal_near_obstacle(occupied_bounded, device)
                raw_env._desired_pos_w[eid] = goal
                steps_on_goal[eid] = 0
                continue

            pos = raw_env._robot.data.root_pos_w[eid]
            goal_local = raw_env._desired_pos_w[eid]
            origin = raw_env._terrain.env_origins[eid]
            pos_local = pos.clone()
            pos_local[:2] -= origin[:2]
            dist = torch.linalg.norm(pos_local - goal_local).item()

            reached = dist < goal_dist_thresh
            timed_out = steps_on_goal[eid].item() > goal_timeout

            if reached or timed_out:
                # Build local map and check occupancy content
                combined = raw_env._build_local_combined_map(eid)
                occ_channel = combined[0]  # (8, 16, 16)
                n_occ = (occ_channel > 0.5).sum().item()

                if n_occ >= min_occ:
                    path = os.path.join(output_dir, f"sample_{total_saved:06d}.npy")
                    np.save(path, combined.cpu().numpy())
                    total_saved += 1
                    if reached:
                        total_reached += 1
                    else:
                        total_timeout += 1
                else:
                    total_skipped_empty += 1

                # Pick next goal near obstacle
                goal = pick_goal_near_obstacle(occupied_bounded, device)
                raw_env._desired_pos_w[eid] = goal
                steps_on_goal[eid] = 0

                if (total_saved - existing) >= target:
                    break

        if step_count % 500 == 0:
            n = total_saved - existing
            print(f"  step {step_count}: {n}/{target} samples "
                  f"(reached={total_reached}, timeout={total_timeout}, "
                  f"skipped_empty={total_skipped_empty})")

    n = total_saved - existing
    print(f"\n[DONE] Collected {n} samples "
          f"(reached={total_reached}, timeout={total_timeout}, "
          f"skipped_empty={total_skipped_empty})")
    print(f"  Total in {output_dir}: {total_saved}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
