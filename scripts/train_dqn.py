import sys
import os
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.algorithms.qlearning_ai import GomokuEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def load_config():
    config_path = os.path.join("config", "algorithms.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = load_config()
    dqn_config = config.get("algorithms", {}).get("qlearning", {})
    model_path = dqn_config.get("model_path", "models/dqn_gomoku")
    
    total_timesteps = 1_000_000
    continue_training = True

    env = GomokuEnv(board_size=15)
    
    model = None
    
    if continue_training and os.path.exists(model_path + ".zip"):
        print(f"Loading model: {model_path}.zip")
        model = DQN.load(model_path, env=env, device=device)
    else:
        print("Creating new model...")
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1,
            device=device,
            batch_size=256,
            buffer_size=100000,
            learning_rate=dqn_config.get("learning_rate", 1e-4),
            learning_starts=5000,
            gamma=dqn_config.get("gamma", 0.99),
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            tensorboard_log="./data/logs/dqn_tensorboard/"
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./data/models/checkpoints/',
        name_prefix='dqn_run'
    )
    
    print(f"Training for {total_timesteps} steps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            reset_num_timesteps=not continue_training 
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    model.save(model_path)
    print(f"Model saved: {model_path}.zip")

if __name__ == "__main__":
    train()
