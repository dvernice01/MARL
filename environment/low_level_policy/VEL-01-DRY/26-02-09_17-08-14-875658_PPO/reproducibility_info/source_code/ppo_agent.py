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
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Network definition matching the yaml config: [256, 256]
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions) # Mean output
        )
        
        # Log STD parameter (learnable)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), initial_log_std, dtype=torch.float32))

    def compute(self, inputs, role):
        # inputs["states"] has shape (batch_size, num_observations)
        x = inputs["states"]
        return self.net(x), self.log_std_parameter, {}

class VelocityControllerValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Network definition matching the yaml config for value: [256, 128]
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1) # Value output
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
        "directory": "", # Placeholder, set in factory
        "experiment_name": "", # Placeholder
        "write_interval": 100,
        "checkpoint_interval": 10000
    }
}

def get_ppo_agent(env, device, agent_cfg=None, log_dir="logs/defaults"):
    """
    Factory function to instantiate a PPO agent with manual model configuration.
    
    Args:
        env: The Gym environment (wrapped).
        device: Torch device.
        agent_cfg: Optional overrides.
        log_dir: Directory for logging.
    
    Returns:
        agent: Instantiated PPO agent.
    """
    
    models = {}
    
    # 1. Instantiate Models
    models = {
        "policy": VelocityControllerPolicy(env.observation_space, env.action_space, device),
        "value": VelocityControllerValue(env.observation_space, env.action_space, device)
    }

    # 2. Agent Configuration
    # Start with SKRL defaults to be safe
    cfg = PPO_DEFAULT_CONFIG.copy()
    
    # Update with OUR explicit defaults
    cfg.update(DEFAULT_PPO_CONFIG)
    
    # If user provided overrides
    if agent_cfg:
        cfg.update(agent_cfg)
        
    # Override logging info
    cfg["experiment"]["directory"] = log_dir
    cfg["experiment"]["experiment_name"] = "" # handled by log_dir path

    # Inject 'size' into preprocessor kwargs which is required by RunningStandardScaler
    if cfg["state_preprocessor_kwargs"] is None:
        cfg["state_preprocessor_kwargs"] = {}
    if cfg["value_preprocessor_kwargs"] is None:
        cfg["value_preprocessor_kwargs"] = {}
        
    cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": device})
    cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

    # 3. Memory
    rollouts = cfg["rollouts"] # Use the value from config
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    # 4. Instantiate Agent
    agent = PPO(models=models, 
                 memory=memory, 
                 cfg=cfg, 
                 observation_space=env.observation_space, 
                 action_space=env.action_space, 
                 device=device)
                 
    return agent
