"""
RL-policy-based 3D local map data collection.

Uses a pre-trained position controller to navigate the drone in the
quadcopter_rnn warehouse environment. Spawns within a restricted region
and only saves maps that contain occupancy.

Usage (from environment/quadcopter_rnn/):
  python scripts/skrl/collect_dataset.py --task=Template-Quadcopter-Rnn-Direct-v0 --checkpoint=/workspace/environment/quadcopter_hierarchical_control/runs/manual_run/cosmic-smoke-232/26-05-18_14-18-46-704539_PPO/checkpoints/best_agent.pt --num_envs=3 --target_samples=1000 --seed=-1 --headless
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
parser.add_argument("--min_svs_voxels", type=int, default=5,
                    help="Min nonzero cells in the SVS channel for a sample to count as SVS-informative")
parser.add_argument("--max_poor_samples", type=int, default=20,
                    help="Max number of SVS-poor samples to keep (e.g. 20 per 1000); "
                         "once reached, further SVS-poor samples are skipped")
parser.add_argument("--respawn_every", type=int, default=10,
                    help="Hops a drone chains (accumulating visit counts → richer SVS) "
                         "before a forced respawn that wipes counts. 1 = respawn every "
                         "hop (poor SVS); larger = richer SVS, more OCC spatial correlation")
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
import hashlib
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
from isaaclab.utils.math import subtract_frame_transforms

rnn_mod.LOCAL_MAP_SAVE_EVERY = 0
LOCAL_HALF_X = (rnn_mod.LOCAL_NX / 2.0) * rnn_mod.LOCAL_CELL_SIZE   # 2.0 m
LOCAL_HALF_Y = (rnn_mod.LOCAL_NY / 2.0) * rnn_mod.LOCAL_CELL_SIZE   # 2.0 m
LOCAL_HALF_Z = (rnn_mod.LOCAL_NZ / 2.0) * rnn_mod.LOCAL_CELL_SIZE   # 1.0 m

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

def pick_goal_in_local_map(raw_env, eid, device):
    """Random goal inside the drone's local-map box, returned in WORLD frame.
    Clamped to the restricted warehouse region (warehouse frame)."""
    pos_w = raw_env._robot.data.root_pos_w[eid]
    origin = raw_env._terrain.env_origins[eid]

    pos_wh = pos_w.clone()
    pos_wh[:2] -= origin[:2]                       # drone pos in warehouse frame

    off = torch.empty(3, device=device)
    off[0] = torch.empty(1, device=device).uniform_(-LOCAL_HALF_X, LOCAL_HALF_X)
    off[1] = torch.empty(1, device=device).uniform_(-LOCAL_HALF_Y, LOCAL_HALF_Y)
    off[2] = torch.empty(1, device=device).uniform_(-LOCAL_HALF_Z, LOCAL_HALF_Z)

    goal_wh = pos_wh + off
    goal_wh[0] = torch.clamp(goal_wh[0], X_MIN + MARGIN, X_MAX - MARGIN)
    goal_wh[1] = torch.clamp(goal_wh[1], Y_MIN + MARGIN, Y_MAX - MARGIN)
    goal_wh[2] = torch.clamp(goal_wh[2], Z_MIN + MARGIN, Z_MAX - MARGIN)

    goal_w = goal_wh.clone()
    goal_w[:2] += origin[:2]                       # back to world frame
    return goal_w


ARENA_SIZE = 4.0   # value the hierarchical-control policy was trained with


def build_hc_observation(raw_env, prev_actions):
    """Observation in the quadcopter_hierarchical_control layout:
    [prev_actions(4), rel_pos_b(3), lin_vel_b(3), ang_vel_b(3), bounds_x(1), bounds_y(1)] = 15

    bounds_x/y are fed a constant 'safely centered' value (ARENA_SIZE/2): the
    exploration walk deliberately roams far from each env origin, which is out
    of distribution for the trained bounds signal (where negative meant 'dying').
    """
    rel_pos_b, _ = subtract_frame_transforms(
        raw_env._robot.data.root_pos_w,
        raw_env._robot.data.root_quat_w,
        raw_env._desired_pos_w,
    )
    n = raw_env._robot.data.root_pos_w.shape[0]
    safe_bound = torch.full((n, 1), ARENA_SIZE / 2.0, device=raw_env.device)

    return torch.cat(
        [
            prev_actions,                          # 4
            rel_pos_b,                             # 3
            raw_env._robot.data.root_lin_vel_b,    # 3
            raw_env._robot.data.root_ang_vel_b,    # 3
            safe_bound,                            # 1  bounds_x
            safe_bound,                            # 1  bounds_y
        ],
        dim=-1,
    )



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
            env.observation_space, env.action_space, device, hidden_size=256
        ),
        "value": VelocityControllerValue(
            env.observation_space, env.action_space, device, hidden_size=256
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
    min_svs = args_cli.min_svs_voxels
    max_poor = args_cli.max_poor_samples
    respawn_every = args_cli.respawn_every

    obs, _ = env.reset()

    # Set initial goals: random point in each drone's local-map box
    for eid in range(num_envs):
        goal = pick_goal_in_local_map(raw_env, eid, device)
        raw_env._desired_pos_w[eid] = goal

    prev_actions = torch.zeros(num_envs, 4, device=device)
    obs = build_hc_observation(raw_env, prev_actions)

    # Dedup: never save two samples with an identical occupancy channel.
    # Seed with existing samples so re-runs don't re-add duplicates.
    saved_occ_hashes = set()
    total_poor_saved = 0
    for f in [x for x in os.listdir(output_dir) if x.endswith(".npy")]:
        a = np.load(os.path.join(output_dir, f))
        saved_occ_hashes.add(hashlib.md5(np.ascontiguousarray(a[0]).tobytes()).hexdigest())
        if int((a[1] > 0).sum()) < min_svs:
            total_poor_saved += 1

    steps_on_goal = torch.zeros(num_envs, device=device)
    hops_since_respawn = [0] * num_envs   # chained hops since last visit-count wipe
    total_saved = existing
    total_reached = 0
    total_timeout = 0
    total_skipped_empty = 0
    total_skipped_dup = 0
    total_skipped_poor = 0
    step_count = 0

    print(f"[INFO] Collecting {target} new samples (min {min_occ} occ voxels per sample)...")

    # Whole loop runs under inference_mode: this is a pure data-collection
    # script (no autograd), and the manual _reset_idx below mutates env
    # buffers that the env's own step()-time reset treats as inference
    # tensors — both must run in the same context.
    while (total_saved - existing) < target and simulation_app.is_running():
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            _, _, terminated, truncated, _ = env.step(actions)
            prev_actions = actions.clone()
            obs = build_hc_observation(raw_env, prev_actions)

            steps_on_goal += 1
            step_count += 1

            # Set True whenever an env's goal changed or it was respawned this
            # iteration → its obs is stale and must be rebuilt before next act().
            did_reset = False

            for eid in range(num_envs):
                is_term = bool(terminated[eid]) if terminated is not None else False
                is_trunc = bool(truncated[eid]) if truncated is not None else False

                # Env auto-resets a done env inside env.step() (this also wiped
                # its visit counts); restart the hop chain and pick a fresh
                # goal relative to its new spawn position.
                if is_term or is_trunc:
                    hops_since_respawn[eid] = 0
                    goal = pick_goal_in_local_map(raw_env, eid, device)
                    raw_env._desired_pos_w[eid] = goal
                    steps_on_goal[eid] = 0
                    did_reset = True
                    continue

                pos_w = raw_env._robot.data.root_pos_w[eid]
                goal_w = raw_env._desired_pos_w[eid]
                dist = torch.linalg.norm(pos_w - goal_w).item()
                if step_count % 500 == 0:
                    print(f"distance error (eid={eid}):", dist)

                reached = dist < goal_dist_thresh
                timed_out = steps_on_goal[eid].item() > goal_timeout

                if reached or timed_out:
                    combined = raw_env._build_local_combined_map(eid)
                    occ_channel = combined[0]
                    n_occ = (occ_channel > 0.5).sum().item()

                    if n_occ >= min_occ:
                        occ_np = np.ascontiguousarray(occ_channel.cpu().numpy())
                        occ_key = hashlib.md5(occ_np.tobytes()).hexdigest()
                        if occ_key in saved_occ_hashes:
                            total_skipped_dup += 1
                        else:
                            # SVS-poor = too few nonzero cells in the SVS
                            # channel. Keep some (richer dataset) but cap them
                            # at max_poor; don't record the occ hash on a
                            # quota-skip so a richer-SVS version of the same
                            # crop can still be saved later.
                            n_svs = int((combined[1] > 0).sum().item())
                            is_poor = n_svs < min_svs
                            if is_poor and total_poor_saved >= max_poor:
                                total_skipped_poor += 1
                            else:
                                saved_occ_hashes.add(occ_key)
                                path = os.path.join(output_dir, f"sample_{total_saved:06d}.npy")
                                np.save(path, combined.cpu().numpy())
                                total_saved += 1
                                if is_poor:
                                    total_poor_saved += 1
                                if reached:
                                    total_reached += 1
                                else:
                                    total_timeout += 1
                    else:
                        total_skipped_empty += 1

                    # Chain hops so global_visit_counts accumulates → the SVS
                    # crop reflects real local exploration, not one ~2 m stub.
                    # Force a respawn (which wipes visit counts) only every
                    # `respawn_every` hops, so the drone still covers different
                    # warehouse regions for OCC variety.
                    hops_since_respawn[eid] += 1
                    if hops_since_respawn[eid] >= respawn_every:
                        raw_env._reset_idx(torch.tensor([eid], device=device))
                        hops_since_respawn[eid] = 0
                    goal = pick_goal_in_local_map(raw_env, eid, device)
                    raw_env._desired_pos_w[eid] = goal
                    steps_on_goal[eid] = 0
                    did_reset = True

                    if (total_saved - existing) >= target:
                        break

            # Rebuild obs so the next agent.act() sees updated goals/positions.
            if did_reset:
                obs = build_hc_observation(raw_env, prev_actions)

            if step_count % 500 == 0:
                n = total_saved - existing
                print(f"  step {step_count}: {n}/{target} samples "
                      f"(reached={total_reached}, timeout={total_timeout}, "
                      f"poor_svs={total_poor_saved}/{max_poor}, "
                      f"skipped_empty={total_skipped_empty}, "
                      f"skipped_dup={total_skipped_dup}, "
                      f"skipped_poor={total_skipped_poor})")

    n = total_saved - existing
    print(f"\n[DONE] {n}/{target} new unique samples "
          f"(reached={total_reached}, timeout={total_timeout}, "
          f"poor_svs={total_poor_saved}/{max_poor}, "
          f"skipped_empty={total_skipped_empty}, "
          f"skipped_dup={total_skipped_dup}, "
          f"skipped_poor={total_skipped_poor})")
    print(f"  Total in {output_dir}: {total_saved}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()