# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

MARL research project for quadcopter navigation in a warehouse simulation using NVIDIA Isaac Lab (Isaac Sim physics). The primary environment under active development is `quadcopter_rnn`.

## Commands

All commands must be run from inside the specific environment directory (e.g. `environment/quadcopter_rnn/`). Isaac Lab must be installed and active (`isaaclab.sh -p` can replace `python` if not in a venv/conda).

**Install an environment (once, editable):**
```bash
cd environment/quadcopter_rnn
python -m pip install -e source/quadcopter_rnn
```

**List registered tasks:**
```bash
python scripts/list_envs.py
```

**Train:**
```bash
python scripts/skrl/train.py --task=Template-Quadcopter-Rnn-Direct-v0 --num_envs=4
python scripts/skrl/train.py --task=Template-Quadcopter-Rnn-Direct-v0 --num_envs=4 --checkpoint=<path/to/best_agent.pt>
```

**Play / evaluate a checkpoint:**
```bash
python scripts/skrl/play.py --task=Template-Quadcopter-Rnn-Direct-v0 --checkpoint=<path>
```

**Dummy agents (verify env setup):**
```bash
python scripts/zero_agent.py --task=Template-Quadcopter-Rnn-Direct-v0
python scripts/random_agent.py --task=Template-Quadcopter-Rnn-Direct-v0
```

**Code formatting:**
```bash
pre-commit run --all-files
```

**VAE training (offline, no simulator):**
```bash
cd vae_container/Vae
python train_no_agent.py
python training_offline_sweep.py   # W&B hyperparameter sweep
```

## Architecture

### Directory layout

```
Marl_IsaacLab/
├── IsaacLab/              # Isaac Lab framework (git-ignored, must be installed separately)
├── environment/           # All RL environments
│   ├── quadcopter_rnn/            ← primary active env
│   ├── quadcopter_vel_control/    ← low-level velocity controller (pre-trained)
│   ├── QuadcopterVae/             ← VAE-based perception variant
│   ├── quadcopter_hierarchical_control/
│   ├── quadcopter/
│   └── low_level_policy/
└── vae_container/Vae/     # Offline VAE training (depth images → latent)
```

### Every environment has the same internal structure

```
environment/<env>/
├── scripts/skrl/
│   ├── train.py           # entry point for training
│   └── play.py            # entry point for inference
└── source/<env>/<env>/
    ├── __init__.py        # gym.register() call — defines the task ID string
    └── tasks/direct/<env>/
        ├── <env>_env.py       # DirectRLEnv subclass: all RL logic
        ├── <env>_env_cfg.py   # @configclass: all hyperparameters
        └── agents/
            └── skrl_ppo_cfg.yaml   # PPO network architecture and training config
```

### quadcopter_rnn architecture (two-level control)

The env implements **hierarchical control**:
1. **High-level policy** (trained here): outputs 4D action `[vx, vy, vz, yaw_rate]` scaled by `max_velocity` / `max_yaw_rate`.
2. **Low-level policy** (pre-trained, frozen): a 13→4 MLP loaded from `quadcopter_vel_control`, called every `decimation_low_level=2` physics steps. It translates velocity commands into thrust/moment.

The low-level policy checkpoint path and its `RunningStandardScaler` are hardcoded in `_load_policy_network()`.

### 3D occupancy map + SVS map

Two spatial data structures maintained per environment:

| Structure | Shape | Purpose |
|---|---|---|
| `global_occ_map` | `(186, 1496, 720)` | Pre-built warehouse voxel map, 0.05m/cell. Values: 0=unknown, 1=free, 2=occupied. Shared across all envs (read-only). |
| `global_visit_counts` | `(num_envs, 186, 1496, 720)` | Per-env visit count per voxel. Reset on episode end. |

At each observation step, a local `(2, NZ, NY, NX)` map is cropped around the drone and optionally saved:
- **Channel 0**: binary occupancy (max-pool from global map)
- **Channel 1**: Shannon entropy of visit distribution (SVS, sum-pool from visit counts)

Key constants controlling this (top of `quadcopter_rnn_env.py`):
- `LOCAL_NZ/NY/NX` — voxel resolution (currently 8×16×16)
- `LOCAL_CELL_SIZE` — meters per local voxel; governs how many global 0.05m voxels are pooled per local voxel
- `LOCAL_MAP_SAVE_EVERY` — global steps between saves to disk (`LOCAL_MAPS_SAVE_DIR`)
- `LOCAL_MAP_START_STEP` — global step at which saving begins

### Path conventions

All hardcoded paths in env code use `/workspace/...` (Docker container paths). Outside the container, the equivalent root is `/home/studenti/dvernice/Marl_IsaacLab/`. The occupancy map lives at `environment/quadcopter_rnn/warehouse_3d_denser/`.

### Reward structure

Rewards are defined in `_get_rewards()` in `<env>_env.py` and scaled by parameters in `<env>_env_cfg.py`. All reward terms are multiplied by `step_dt` to be time-invariant. The cfg file is the single source of truth for all scales.

### W&B integration

`wandb.login()` is called at module import. All training metrics are logged automatically by SKRL. The VAE uses a separate W&B sweep via `config.yaml`.

### Code implementation
Everytime there is something to change in the code write it in the chat. Don't change it directly or ask me to chenge it in my place. 

Write just things you are really sure of.

Ask me everytime you have doubts on something. Don't try to guess. I prefer to spend more time before having the answer rather than having an imprecise answer. For this reason take your time to elaborate the response.

I write on python flies so write code with the correct indendation. Something that allow me just to make copy and paste without need to fix the indendation.

