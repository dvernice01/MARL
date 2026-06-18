# DOCKER SECTION
Firstly, open IsaacLab_reduced folder and run the following command:

    - docker pull nvcr.io/nvidia/isaac-lab:2.3.2 

Make it exacutable with:

    - chmod +x run.sh

and run it with:

    - ./run.sh

# PROJECT TREE
```
├── IsaacLab_reduced/                          # For IsaacLab container installation
├── environment/                               # RL IsaacLab environment 
│   ├── QuadcopterVae/                         # VAE-conditioned quadcopter env
│   ├── quadcopter_hierarchical_control/       # Hierarchical control env (position controller in free-space without sensors)
│   ├── quadcopter_rnn/                        # RNN-based quadcopter env
│   ├── quadcopter_vel_control/                # Velocity control env
│   └── uav_navigation/                        # Full pipeline for position control in complex env
├── vae_container/                             # Containerized VAE pipeline
    ├── Vae/                                   # VAE model & training code
    │   ├── autoencoder3D.py                   # 3D autoencoder architecture
    │   ├── config.yaml                        # VAE training config
    │   ├── config_3d.yaml                     # 3D autoencoder config
    │   ├── occupancymap3d_fw.py               # 3D occupancy map generator
    │   ├── training_3D_autoencoder.py         # 3D-AE training loop (Wandb)
    │   ├── training_offline_sweep.py          # VAe training loop (Wandb)
    │   └── vae_residual_batch.py              # VAE architecture
    ├── docker/                                # Docker configuration files

```

# Short explaination of the folder utility

    - quadcopter_vel_control -> is the project used for the training of the low-level control, that is the velocity control (python scripts/skrl/train_manual.py --task=Template-Quadcopter_Vel_Control-Direct-v0 --enable_cameras --headless)
    
    - quadcopter_hierarchical_control -> is the for position controller that use the velocity control to reach desired position in the env (python scripts/skrl/train_manual.py --task=Template-Quadcopter_Hierarchical_Control-Direct-v0 --enable_cameras --headless)
    
    - quadcopter_rnn -> Firstly it was used to test the RNN implementation, then it has become the environment used to collect OCC+SVS maps. The command to run to get maps is: python scripts/skrl/collect_occ_svs_dataset.py --task=Template-Quadcopter-Rnn-Direct-v0 --checkpoint=/workspace/environment/quadcopter_hierarchical_control/runs/manual_run/cosmic-smoke-232/26-05-18_14-18-46-704539_PPO/checkpoints/best_agent.pt --num_envs=3 --target_samples=1000 --seed=-1 --headless

    - QuadcopterVae -> It contains the setup to test validity of the VAE in simulation and it was used to collect depth dataset for VAE training. The command to run to collect dataset is: python source/QuadcopterVae/QuadcopterVae/tasks/direct/quadcoptervae/collect_depth_dataset.py --num_envs=20  --num_samples=5000 --output_dir=/workspace/vae_container/Vae/isaaclab_dataset_right_size   --enable_cameras  --headless

    - uav_navigation -> full pipeline. See below.

    - vae_container -> Container used for ML section. It contains the VAE and 3D-AE architectures (vae_residual_batch.py and autoencoder3d.py) and the training file runnable by wandb agent:
          - wandb sweep --project vae_training config.yaml or wandb sweep --project 3d_ae_training config_3d.yaml and then by copying and pasting the yellow sentence

# RL PROJECT SECTION
All the RL projects in the environment folder share the same architecture with same files. For this reason, the structure is created for uav_navigation file but it fits every project.

```
└── uav_navigation/                                     # UAV navigation workspace
    ├── scripts/
    │   └── skrl/                                       # SKRL-based training scripts
    │       ├── agents/
    │       │   └── ppo_agent.py                        # PPO agent definition
    │       └── train_manual.py                         # Manual training entry point
    │       └── play_sequential.py                      # Script to run the trained policy
    └── source/
        └── uav_navigation/
            └── uav_navigation/
                └── tasks/
                    └── direct/                         # Direct task implementations
                        ├── vae_residual_batch.py       # VAE architecture
                        ├── autoencoder3D.py            # 3D-AE architecture
                        ├── uav_navigation_env.py       # Environment definition
                        └── uav_navigation_env_cfg.py   # Environment configuration
```
The environments is runnable by using the following commands:
  - cd /workspace/environment/uav_navigation
  - python -m pip install -e source/uav_navigation
  - pip install skrl==1.4.3
  - python scripts/skrl/train_manual.py --task=Template-Uav-Navigation-Direct-v0 --enable_cameras --headless


