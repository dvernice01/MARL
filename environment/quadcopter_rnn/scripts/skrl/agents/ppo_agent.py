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
                 num_envs=1, num_layers=1, hidden_size=256, hidden_size_gru=512, sequence_length=8):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.num_envs        = num_envs
        self.num_layers      = num_layers
        self.hidden_size     = hidden_size      # MLP output size
        self.hidden_size_gru = hidden_size_gru  # GRU hidden size
        self.sequence_length = sequence_length
        pure_obs_dim = self.num_observations - self.num_actions

        self.net = nn.Sequential(
            nn.Linear(pure_obs_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )

        self.gru = nn.GRU(
            input_size=self.num_actions,      
            hidden_size=hidden_size_gru, 
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size + hidden_size_gru, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std, dtype=torch.float32)
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size_gru)]}}

    def compute(self, inputs, role):
        states        = inputs["states"]
        terminated    = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]  # (num_layers, N, hidden_size_gru)

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, full_dim)

            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, self.hidden_size_gru
            )
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (num_layers, N, hidden_size_gru)

            N, L, full_dim = rnn_input.shape

            # split inside the sequence
            pure_obs_seq     = rnn_input[..., :-self.num_actions]   # (N, L, obs_dim)
            prev_actions_seq = rnn_input[..., -self.num_actions:]   # (N, L, num_actions)
            obs_dim          = pure_obs_seq.shape[-1]

            # MLP sees only pure observations — no actions
            mlp_flat = self.net(pure_obs_seq.view(N * L, obs_dim))  # (N*L, hidden_size)
            mlp_out  = mlp_flat.view(N, L, self.hidden_size)         # (N, L, hidden_size)

            # GRU sees only previous actions — no observations
            if terminated is not None and torch.any(terminated):
                rnn_outputs    = []
                terminated_seq = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated_seq[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_out, hidden_states = self.gru(prev_actions_seq[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated_seq[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_out)
                rnn_output = torch.cat(rnn_outputs, dim=1)           # (N, L, hidden_size_gru)
            else:
                rnn_output, hidden_states = self.gru(prev_actions_seq, hidden_states)

            # flatten sequence dim for output head
            mlp_out    = mlp_out.reshape(N * L, self.hidden_size)
            rnn_output = rnn_output.reshape(N * L, self.hidden_size_gru)

        else:
            # rollout — single step
            pure_obs     = states[..., :-self.num_actions]   # (N, obs_dim)
            prev_actions = states[..., -self.num_actions:]   # (N, num_actions)

            mlp_out    = self.net(pure_obs)                              # (N, hidden_size)
            rnn_output, hidden_states = self.gru(
                prev_actions.unsqueeze(1), hidden_states
            )                                                            # (N, 1, hidden_size_gru)
            rnn_output = rnn_output.squeeze(1)                           # (N, hidden_size_gru)

        # combine MLP state features + GRU action memory
        combined = torch.cat([mlp_out, rnn_output], dim=-1)              # (N*L, hidden_size + hidden_size_gru)

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.tanh(x), self.log_std_parameter, {"rnn": [hidden_states]}


class HierarchicalGRUValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=256, hidden_size_gru=512, sequence_length=8):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs        = num_envs
        self.num_layers      = num_layers
        self.hidden_size     = hidden_size
        self.hidden_size_gru = hidden_size_gru
        self.sequence_length = sequence_length

        pure_obs_dim = self.num_observations - self.num_actions  # same split as policy

        # MLP: pure obs only, no actions
        self.net = nn.Sequential(
            nn.Linear(pure_obs_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )

        # GRU: only prev_actions
        self.gru = nn.GRU(
            input_size=self.num_actions,
            hidden_size=hidden_size_gru,
            num_layers=num_layers,
            batch_first=True,
        )

        # fc1 must match combined size
        self.fc1 = nn.Linear(hidden_size + hidden_size_gru, 64)  # ← 256+512=768
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # value is scalar

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size_gru)]}}

    def compute(self, inputs, role):
        states        = inputs["states"]
        terminated    = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, self.hidden_size_gru
            )
            hidden_states = hidden_states[:, :, 0, :].contiguous()

            N, L, full_dim = rnn_input.shape

            # split
            pure_obs_seq     = rnn_input[..., :-self.num_actions]  # (N, L, pure_obs_dim)
            prev_actions_seq = rnn_input[..., -self.num_actions:]  # (N, L, num_actions)
            obs_dim          = pure_obs_seq.shape[-1]

            # MLP
            mlp_flat = self.net(pure_obs_seq.view(N * L, obs_dim))  # (N*L, hidden_size)
            mlp_out  = mlp_flat.view(N, L, self.hidden_size)

            # GRU
            if terminated is not None and torch.any(terminated):
                rnn_outputs    = []
                terminated_seq = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated_seq[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_out, hidden_states = self.gru(prev_actions_seq[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated_seq[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_out)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(prev_actions_seq, hidden_states)

            mlp_out    = mlp_out.reshape(N * L, self.hidden_size)
            rnn_output = rnn_output.reshape(N * L, self.hidden_size_gru)

        else:
            pure_obs     = states[..., :-self.num_actions]
            prev_actions = states[..., -self.num_actions:]

            mlp_out    = self.net(pure_obs)
            rnn_output, hidden_states = self.gru(
                prev_actions.unsqueeze(1), hidden_states
            )
            rnn_output = rnn_output.squeeze(1)

        # combine and compute value
        combined = torch.cat([mlp_out, rnn_output], dim=-1)  # (N*L, hidden_size + hidden_size_gru)

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                    # (N*L, 1)

        # NO tanh for value function — value is unbounded
        return x, {"rnn": [hidden_states]} # ← removed torch.tanh




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
            "project": "quadcopter_rnn",  # Must be a string
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
    hidden_size_gru    = get("hidden_size_gru", 512)
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


