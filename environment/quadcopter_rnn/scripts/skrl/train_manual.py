
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train Velocity Controller RL agent with skrl using MANUAL instantiation.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import random
import json
import shutil
import subprocess
from datetime import datetime
import wandb

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Velocity Controller with skrl defined manually.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Quadcopter-Rnn-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Name of the experiment to run or resume. If provided, the timestamp will not be appended.",
)
parser.add_argument("--agent", type=str, default="ppo", help="Agent type (ppo).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import skrl
from skrl.trainers.torch import SequentialTrainer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.utils.io import dump_yaml

import quadcopter_rnn.tasks.direct.quadcopter_rnn # noqa: F401


# config shortcuts
agent_cfg_entry_point = "skrl_cfg_entry_point"

def save_reproducibility_info(log_dir, agent, args_cli, env_cfg):
    """Saves metadata, network architecture, and source code for reproducibility."""
    repro_dir = os.path.join(log_dir, "reproducibility_info")
    os.makedirs(repro_dir, exist_ok=True)
    
    # 1. Metadata JSON (Git info, args, config)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "args_cli": vars(args_cli),
        # We try to extract git info
        "git_commit": "unknown",
        "git_dirty": False,
        "agent_config": agent.cfg,
        # env_cfg is complex object, maybe just save key values if needed.
        "env_seed": getattr(env_cfg, "seed", "unknown")
    }
    
    try:
        metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        metadata["git_dirty"] = bool(diff)
    except Exception as e:
        print(f"[WARNING] Could not retrieve git info: {e}")

    with open(os.path.join(repro_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, default=lambda o: str(o))

    # 2. Model Architecture
    with open(os.path.join(repro_dir, "model_architecture.txt"), "w") as f:
        f.write("Policy Network:\n")
        f.write(str(agent.policy))
        f.write("\n\nValue Network:\n")
        f.write(str(agent.value))

    # 3. Source Code Snapshot
    source_dir = os.path.join(repro_dir, "source_code")
    os.makedirs(source_dir, exist_ok=True)
    
    files_to_copy = [
        # Script itself
        os.path.abspath(__file__),
        # Agent Factory (Dynamically copy based on active agent?)
        # For now we copy the one used.
        # Env
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/quadcopter_rnn/quadcopter_rnn/tasks/direct/quadcopter_rnn/quadcopter_rnn_env.py")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/quadcopter_rnn/quadcopter_rnn/tasks/direct/quadcopter_rnn/quadcopter_rnn_env_cfg.py")),
    ]
    
    # Add Agent Config File
    if args_cli.agent == "ppo":
         files_to_copy.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "ppo_agent.py"))
    
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            shutil.copy2(file_path, source_dir)
        else:
            print(f"[WARNING] Could not backup source file: {file_path}")

    print(f"[INFO] Reproducibility info saved to: {repro_dir}")

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):

    # ── 1. Init wandb FIRST so sweep config is available ──────────────
    run = wandb.init(
        project="quadcopter_rnn",
        sync_tensorboard=True,
    )
    sweep_cfg = wandb.config  # sweep controller injects values here

    # ── 2. Apply reward scales from sweep to env_cfg ───────────────────
    def apply_if_present(key, target_obj, negate=False):
        if key in sweep_cfg:
            val = sweep_cfg[key]
            setattr(target_obj, key, -val if negate else val)

    apply_if_present("lin_vel_reward_scale",       env_cfg)
    apply_if_present("ang_vel_reward_scale",        env_cfg)
    apply_if_present("distance_to_goal_reward_scale", env_cfg)
    apply_if_present("alive_reward_scale",          env_cfg)
    # death_reward_scale uses the abs trick — negate here
    if "death_reward_scale_abs" in sweep_cfg:
        env_cfg.death_reward_scale = -sweep_cfg["death_reward_scale"]

    # ── 3. Apply env/sim overrides ─────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device     = args_cli.device   if args_cli.device   is not None else env_cfg.sim.device
    if args_cli.seed == -1:
        import random
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed           = args_cli.seed      if args_cli.seed     is not None else env_cfg.seed

    experiment_name = args_cli.experiment_name if args_cli.experiment_name \
        else datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Manual_{args_cli.agent.upper()}"
    log_dir = os.path.join("projects", "quadcopter_rnn", "logs", "skrl",
                           "quadcopter_rnn_direct", experiment_name)

    # ── 4. Create environment ──────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── 5. Create agent — pass sweep_cfg so it reads lr, hidden_size etc ──
    device = env.device
    if args_cli.agent == "ppo":
        from agents.ppo_agent import get_ppo_agent
        agent_factory = get_ppo_agent
    else:
        raise ValueError(f"Agent '{args_cli.agent}' not supported.")

    agent = agent_factory(
        env=env,
        device=device,
        agent_cfg=sweep_cfg,   # ← sweep config flows in here
        log_dir=log_dir
    )

    # ── 6. Trainer ────────────────────────────────────────────────────
    rollouts = sweep_cfg.get("rollouts", 64)
    default_timesteps = 150000
    trainer_cfg = {
        "timesteps": args_cli.max_iterations * rollouts if args_cli.max_iterations else default_timesteps,
        "headless": True,
        "close_environment_at_exit": False,
        "environment_info": "log",
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
