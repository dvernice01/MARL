import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler

# --- Custom Model Definitions ---
class HierarchicalGRUPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=0,
                 num_envs=1, num_layers=1, hidden_size=2048, hidden_size_gru=512, sequence_length=8):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.num_envs        = num_envs
        self.num_layers      = num_layers
        self.hidden_size     = hidden_size      # MLP output size
        self.hidden_size_gru = hidden_size_gru  # GRU hidden size
        self.sequence_length = sequence_length

        # MLP: obs → hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ELU(),
        )

        # GRU: takes MLP output (hidden_size) → hidden_size_gru
        self.gru = nn.GRU(
            input_size=hidden_size // 4,      # MLP output feeds into GRU
            hidden_size=hidden_size_gru, # GRU internal size
            num_layers=num_layers,
            batch_first=True,
        )

        # Head: hidden_size_gru → actions
        self.fc1 = nn.Linear(hidden_size_gru, hidden_size_gru // 2)
        self.fc2 = nn.Linear(hidden_size_gru // 2, hidden_size_gru // 8)
        self.fc3 = nn.Linear(hidden_size_gru // 8, self.num_actions)

        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std, dtype=torch.float32)
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size_gru)]}}

    def compute(self, inputs, role):
        states         = inputs["states"]
        terminated     = inputs.get("terminated", None)
        hidden_states  = inputs["rnn"][0]  # (num_layers, N, hidden_size_gru)

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, obs_dim)

            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, self.hidden_size_gru
            )
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (num_layers, N, hidden_size_gru)

            N, L, obs_dim   = rnn_input.shape
            features_flat   = self.net(rnn_input.view(N * L, obs_dim))  # (N*L, hidden_size // 4)
            features        = features_flat.view(N, L, self.hidden_size // 4)  # (N, L, hidden_size // 4)

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated  = terminated.view(-1, self.sequence_length)
                indexes     = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(features[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(features, hidden_states)

        else:
            features              = self.net(states).unsqueeze(1)         # (N, 1, hidden_size)
            rnn_output, hidden_states = self.gru(features, hidden_states) # (N, 1, hidden_size_gru)

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)   # (N*L, hidden_size_gru)

        x = F.relu(self.fc1(rnn_output))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [hidden_states]}


class HierarchicalGRUValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=2048, hidden_size_gru=512, sequence_length=8):

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs        = num_envs
        self.num_layers      = num_layers
        self.hidden_size     = hidden_size      # MLP output size
        self.hidden_size_gru = hidden_size_gru  # GRU hidden size
        self.sequence_length = sequence_length

        # MLP: obs → hidden_size
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ELU(),
        )

        # GRU: takes MLP output (hidden_size) → hidden_size_gru
        self.gru = nn.GRU(
            input_size=hidden_size // 4,      # MLP output feeds into GRU
            hidden_size=hidden_size_gru, # GRU internal size
            num_layers=num_layers,
            batch_first=True,
        )

        # Head: hidden_size_gru → 1 (value)
        self.fc1 = nn.Linear(hidden_size_gru, hidden_size_gru // 2)
        self.fc2 = nn.Linear(hidden_size_gru // 2, hidden_size_gru // 8)
        self.fc3 = nn.Linear(hidden_size_gru // 8, 1)

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size_gru)]}}

    def compute(self, inputs, role):
        states        = inputs["states"]
        terminated    = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]  # (num_layers, N, hidden_size_gru)

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, obs_dim)

            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, self.hidden_size_gru
            )
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (num_layers, N, hidden_size_gru)

            N, L, obs_dim = rnn_input.shape
            features_flat = self.net(rnn_input.view(N * L, obs_dim))  # (N*L, hidden_size // 4)
            features      = features_flat.view(N, L, self.hidden_size // 4)  # (N, L, hidden_size // 4)

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated  = terminated.view(-1, self.sequence_length)
                indexes     = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.gru(features[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(features, hidden_states)

        else:
            features              = self.net(states).unsqueeze(1)         # (N, 1, hidden_size)
            rnn_output, hidden_states = self.gru(features, hidden_states) # (N, 1, hidden_size_gru)

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)   # (N*L, hidden_size_gru)

        x = F.relu(self.fc1(rnn_output))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return return self.net(rnn_output), {"rnn": [hidden_states]}


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
            "project": "uav_navigation",  # Must be a string
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
    hidden_size        = get("hidden_size", 2048)
    hidden_size_gru    = get("hidden_size_gru", 512)
    rollouts           = get("rollouts", 256)
    learning_rate      = get("learning_rate", 5e-4)
    learning_epochs    = get("learning_epochs", 15)
    discount_factor    = get("discount_factor", 0.99)
    entropy_loss_scale = get("entropy_loss_scale", 0.0)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update(DEFAULT_PPO_CONFIG)

    # Override with sweep values
    cfg["rollouts"]            = rollouts
    cfg["learning_rate"]       = learning_rate
    cfg["learning_epochs"]     = learning_epochs
    cfg["discount_factor"]     = discount_factor
    cfg["entropy_loss_scale"]  = entropy_loss_scale

    models = {
        "policy": HierarchicalGRUPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs,num_layers=1, hidden_size=hidden_size, hidden_size_gru=hidden_size_gru, sequence_length=8),
        "value":  HierarchicalGRUValue(env.observation_space, env.action_space, device, num_envs=env.num_envs,num_layers=1, hidden_size=hidden_size, hidden_size_gru=hidden_size_gru, sequence_length=8)
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
