# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of the Velocity Controller PPO agent (skrl) and
optionally record a video.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of the Velocity Controller PPO agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="Template-Hierarchical-Controller-Direct-Play-v0", help="Name of the task."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--hidden_size",
    type=int,
    default=None,
    help=(
        "hidden_size used to build VelocityControllerPolicy/Value at training time. "
        "If not given, the script tries to auto-detect it from the run's "
        "reproducibility/config.json (wandb_config.hidden_size). Falls back to 256."
    ),
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

import json
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

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

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

# --- stessi modelli usati in training (agents/ppo_agent.py) ---
from agents.ppo_agent import VelocityControllerPolicy, VelocityControllerValue

import quadcopter_hierarchical_control.tasks  # noqa: F401


agent_cfg_entry_point = "skrl_cfg_entry_point"


def _find_hidden_size(resume_path: str, cli_value: int | None, default: int = 256) -> int:
    """Cerca hidden_size in reproducibility/config.json risalendo dal checkpoint.

    Struttura di training attesa:
        runs/<sweep>/<run_name>/<timestamp>/checkpoints/agent_XXXX.pt
        runs/<sweep>/<run_name>/reproducibility/config.json

    Se non trova nulla, usa cli_value se fornito, altrimenti `default`.
    """
    if cli_value is not None:
        print(f"[INFO] hidden_size forzato da CLI: {cli_value}")
        return cli_value

    current = os.path.abspath(resume_path)
    for _ in range(6):  # risale al massimo 6 livelli di cartelle
        current = os.path.dirname(current)
        candidate = os.path.join(current, "reproducibility", "config.json")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r") as f:
                    data = json.load(f)
                hs = data.get("wandb_config", {}).get("hidden_size", None)
                if hs is not None:
                    print(f"[INFO] hidden_size={hs} letto da {candidate}")
                    return int(hs)
                print(f"[WARNING] '{candidate}' trovato ma senza 'hidden_size' in wandb_config.")
            except Exception as e:
                print(f"[WARNING] Impossibile leggere {candidate}: {e}")
            break  # trovato il file ma senza il dato utile: non serve continuare a risalire

    print(f"[WARNING] Nessun config.json trovato per il checkpoint. Uso hidden_size di default={default}.")
    return default


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
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
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # same as: `wrap_env(env, wrapper="auto")`

    obs_space = env.observation_space
    act_space = env.action_space
    device = env_cfg.sim.device

    # --- hidden_size: deve combaciare con il training ---
    hidden_size = _find_hidden_size(resume_path, args_cli.hidden_size, default=256)

    # --- MODELLI (identici a agents/ppo_agent.py) ---
    models = {
        "policy": VelocityControllerPolicy(obs_space, act_space, device, hidden_size=hidden_size),
        "value": VelocityControllerValue(obs_space, act_space, device, hidden_size=hidden_size),
    }

    # --- MEMORIA (serve solo per instanziare l'agente, non usata in play) ---
    memory = RandomMemory(
        memory_size=experiment_cfg["agent"]["rollouts"],
        num_envs=env.num_envs,
        device=device,
    )

    # --- CFG PPO (deve combaciare per via degli stati dei preprocessor salvati nel checkpoint) ---
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
        "state_preprocessor_kwargs": {"size": obs_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "grad_norm_clip": experiment_cfg["agent"]["grad_norm_clip"],
        "ratio_clip": experiment_cfg["agent"]["ratio_clip"],
        "value_clip": experiment_cfg["agent"]["value_clip"],
        "clip_predicted_values": experiment_cfg["agent"]["clip_predicted_values"],
        "entropy_loss_scale": experiment_cfg["agent"]["entropy_loss_scale"],
        "value_loss_scale": experiment_cfg["agent"]["value_loss_scale"],
        "experiment": {
            "write_interval": 0,  # non loggare in play
            "checkpoint_interval": 0,  # non salvare checkpoint in play
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
        device=device,
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
            obs, _, _, _, _ = env.step(actions)

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