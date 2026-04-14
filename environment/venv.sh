cd quadcopter_hierarchical_control 
apt update && apt-get install python3.12 python3.12-venv -y
/usr/bin/python3.12 -m venv /workspace/wandb_clean_env
source /workspace/wandb_clean_env/bin/activate
/workspace/wandb_clean_env/bin/pip install wandb