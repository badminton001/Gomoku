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

def train_8x8():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    BOARD_SIZE = 8
    MODEL_PATH = "models/dqn_gomoku_8x8"
    TOTAL_TIMESTEPS = 200_000
    
    config = load_config()
    dqn_config = config.get("algorithms", {}).get("qlearning", {})

    print(f"Creating {BOARD_SIZE}x{BOARD_SIZE} environment...")
    env = GomokuEnv(board_size=BOARD_SIZE)
    
    print(f"Initializing DQN model...")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        device=device,
        batch_size=512,
        buffer_size=50000,
        learning_rate=2e-4,
        learning_starts=1000,
        gamma=0.95,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        tensorboard_log="./data/logs/dqn_8x8_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='./data/models/checkpoints_8x8/',
        name_prefix='dqn_8x8'
    )
    
    print(f"Starting training ({TOTAL_TIMESTEPS} steps)...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            log_interval=100
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    model.save(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}.zip")

if __name__ == "__main__":
    train_8x8()
