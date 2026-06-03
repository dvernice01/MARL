# LA DESCRIZIONE E' RELATIVA SOLO ALLA PIPELINE FINALE, CIOE' IL PROGETTO PER IL CONTROLLO DI POSIZIONE CON VELOCITY CONTROLLER, VAE E 3D-AE INTEGRATI.

# SUCCESSIVAMENTE VERRA' INTEGRATA LA PARTE RELATIVA AGLI ALTRI TRAINING.

I file di interesse sono quelli in riferimento al progetto: "environment/uav_navigation".

Gli scripts per il training seguono la logica di quelli che mi condividesti tu.

Il resto del materiale si trova invece sulla cartella "source".

Troverai quindi:

```
└── uav_navigation/                              # UAV navigation workspace
    ├── scripts/
    │   └── skrl/                                # SKRL-based training scripts
    │       ├── agents/
    │       │   └── ppo_agent.py                 # PPO agent definition
    │       └── train_manual.py                  # Manual training entry point
    └── source/
        └── uav_navigation/
            └── uav_navigation/
                └── tasks/
                    └── direct/                  # Direct task implementations
                        ├── vae_residual_batch.py    # VAE architecture
                        ├── autoencoder3D.py         # 3D-AE architecture
                        ├── uav_navigation_env.py    # Environment definition
                        └── uav_navigation_env_cfg.py # Environment configuration
```

I files "train_manual" e "ppo_agent" sono molto simili ai tuoi. Un cambiamento importante è, però, la presenza della GRU nell'architettura.

I files che descrivono l'architettura delle reti presentano dei commenti pressocchè AI-generated ma sono esattamente gli stessi file usati per il ML-training.

Il file "uav_navigation_env.py" contiene una serie di brevi commenti che spiegano l'utilità delle varie classi. Anche in questo caso la struttura riprende quelle usate durante il ML training però alcune funzioni più lente (sono commentate) sono state sostituite da funzioni più veloci. La classe che rappresenta l'environment sono presenta nulla di troppo particolare tranne che nella funzione "_get_observation" alle lines 1182 e 1225 si possono decommentare le parti commentate per visualizzare in real-time le collision images ricostruite e le SVS+occ maps ricostruite ("local_maps_check.png", "vae2d_check.png").


L'ambiente è runnabile lanciando i seguenti comandi:
  - python -m pip install -e source/uav_navigation
  - pip install skrl==1.4.3
  - python scripts/skrl/train_manual.py --task=Template-Uav-Navigation-Direct-v0 --enable_cameras --headless

Qualora avessi bisogno della chiave wandb è:

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
