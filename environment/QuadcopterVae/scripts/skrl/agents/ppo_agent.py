import os
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler

# --- Custom Model Definitions ---

class VelocityControllerPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256,clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Network definition matching the yaml config: [256, 256]
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.num_actions) # Mean output
        )
        
        # Log STD parameter (learnable)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), initial_log_std, dtype=torch.float32))

    def compute(self, inputs, role):
        # inputs["states"] has shape (batch_size, num_observations)
        x = inputs["states"]
        return self.net(x), self.log_std_parameter, {}

class VelocityControllerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_size=256, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Network definition matching the yaml config for value: [256, 128]
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, 1) # Value output
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        return self.net(x), {}

# --------------------------------
# Explicit Configuration for Manual Training
DEFAULT_PPO_CONFIG = {
    "rollouts": 64,
    "learning_epochs": 15,
    "mini_batches": 8,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 5.0e-04,
    "learning_rate_scheduler": KLAdaptiveLR,
    "learning_rate_scheduler_kwargs": {
        "kl_threshold": 0.008
    },
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": RunningStandardScaler,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 1.0,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": True,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 2.0,
    "kl_threshold": 0.0,
    "rewards_shaper_scale": 0.1,
    "time_limit_bootstrap": False,
    # logging and checkpoint
    "experiment": {
        "directory": "runs",            # Provide a base folder for local logs
        "write_interval": 100,
        "checkpoint_interval": 10000,
        "wandb": True,
        "wandb_kwargs": {
            "project": "QuadcopterVae",  # Must be a string
        }
    }
}


def get_ppo_agent(env, device, agent_cfg=None, log_dir="logs/defaults"):
    
    # Helper to read from wandb.config (sweep) or fall back to default
    def get(key, default):
        if agent_cfg is not None and hasattr(agent_cfg, key):
            return getattr(agent_cfg, key)
        return default

    # Read all sweep parameters with fallbacks to your defaults
    hidden_size        = get("hidden_size", 256)
    rollouts           = get("rollouts", 64)
    learning_rate      = get("learning_rate", 1e-5)
    learning_epochs    = get("learning_epochs", 8)
    discount_factor    = get("discount_factor", 0.99)
    entropy_loss_scale = get("entropy_loss_scale", 0.001)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update(DEFAULT_PPO_CONFIG)

    # Override with sweep values
    cfg["rollouts"]            = rollouts
    cfg["learning_rate"]       = learning_rate
    cfg["learning_epochs"]     = learning_epochs
    cfg["discount_factor"]     = discount_factor
    cfg["entropy_loss_scale"]  = entropy_loss_scale

    models = {
        "policy": VelocityControllerPolicy(env.observation_space, env.action_space, device, hidden_size=hidden_size),
        "value":  VelocityControllerValue(env.observation_space, env.action_space, device, hidden_size=hidden_size)
    }

    cfg["experiment"]["directory"] = log_dir
    cfg["experiment"]["experiment_name"] = ""

    if cfg["state_preprocessor_kwargs"] is None:
        cfg["state_preprocessor_kwargs"] = {}
    if cfg["value_preprocessor_kwargs"] is None:
        cfg["value_preprocessor_kwargs"] = {}

    cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": device})
    cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    return agent


