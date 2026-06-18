# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

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

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# IMPORTATE DA ME ------------------------------------------------
from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler

# stessi modelli GRU usati in training
from agents.ppo_agent import HierarchicalGRUPolicy, HierarchicalGRUValue

# ----------------------------------------------------------------
import torch
import torch.nn as nn

import uav_navigation.tasks  # noqa: F401


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    env_cfg.eval_fixed_spawn_xy = (-10, -16)
    # Make the render/recording camera follow the drone
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (0.0, -6.0, 3.0)    # 6 m behind (-y) and 3 m above the drone
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)  # look at the drone

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    
    obs_space = env.observation_space
    act_space = env.action_space
    device    = env_cfg.sim.device

    # l'architettura deve combaciare con il training (default in agents/ppo_agent.py)
    HIDDEN_SIZE     = 2048
    HIDDEN_SIZE_GRU = 512
    NUM_LAYERS      = 1
    SEQUENCE_LENGTH = 8

    # --- MODELLI (ricorrenti, identici al training) ---
    models = {}
    models["policy"] = HierarchicalGRUPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        num_envs=env.num_envs,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        hidden_size_gru=HIDDEN_SIZE_GRU,
        sequence_length=SEQUENCE_LENGTH,
    )
    models["value"] = HierarchicalGRUValue(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        num_envs=env.num_envs,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        hidden_size_gru=HIDDEN_SIZE_GRU,
        sequence_length=SEQUENCE_LENGTH,
    )

    # --- MEMORIA ---
    memory = RandomMemory(
        memory_size=experiment_cfg["agent"]["rollouts"],  # <-- era agent_cfg_entry_point
        num_envs=env.num_envs,
        device=env_cfg.sim.device
    )

    # --- CFG PPO ---
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update({
        "rollouts": experiment_cfg["agent"]["rollouts"],
        "learning_epochs": experiment_cfg["agent"]["learning_epochs"],
        "mini_batches": experiment_cfg["agent"]["mini_batches"],
        "discount_factor": experiment_cfg["agent"]["discount_factor"],
        "lambda": experiment_cfg["agent"]["lambda"],
        "learning_rate": experiment_cfg["agent"]["learning_rate"],
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": experiment_cfg["agent"]["learning_rate_scheduler_kwargs"],
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": obs_space, "device": env_cfg.sim.device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": env_cfg.sim.device},
        "grad_norm_clip": experiment_cfg["agent"]["grad_norm_clip"],
        "ratio_clip": experiment_cfg["agent"]["ratio_clip"],
        "value_clip": experiment_cfg["agent"]["value_clip"],
        "clip_predicted_values": experiment_cfg["agent"]["clip_predicted_values"],
        "entropy_loss_scale": experiment_cfg["agent"]["entropy_loss_scale"],
        "value_loss_scale": experiment_cfg["agent"]["value_loss_scale"],
        "experiment": {
            "write_interval": 0,       # <-- no logging in play
            "checkpoint_interval": 0,  # <-- no checkpoint in play
            "directory": "",
        },
    })

    # --- AGENTE ---
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=obs_space,
        action_space=act_space,
        device=env_cfg.sim.device
    )
    agent.init()

    # --- CARICA CHECKPOINT ---
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)
    agent.set_running_mode("eval")

    # --- LOOP DI PLAY ---
    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

            # avanza lo stato nascosto del GRU e lo azzera per gli env terminati
            # (in un loop manuale questo avviene normalmente in agent.record_transition)
            if getattr(agent, "_rnn", False):
                agent._rnn_initial_states = agent._rnn_final_states
                done = (terminated | truncated).nonzero(as_tuple=False)
                if done.numel():
                    for s in agent._rnn_initial_states["policy"]:
                        s[:, done[:, 0]] = 0

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
