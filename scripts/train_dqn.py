import sys
import os
import yaml

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
    print(">>> Starting DQN Training...")
    
    config = load_config()
    dqn_config = config.get("algorithms", {}).get("qlearning", {})
    
    model_path = dqn_config.get("model_path", "models/dqn_gomoku")
    total_timesteps = 100_000
    
    env = GomokuEnv(board_size=15)
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=dqn_config.get("learning_rate", 1e-4),
        buffer_size=dqn_config.get("buffer_size", 50000),
        learning_starts=1000,
        batch_size=32,
        gamma=dqn_config.get("gamma", 0.99),
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        tensorboard_log="./data/logs/dqn_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./data/models/checkpoints/',
        name_prefix='dqn_model'
    )
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
        
    model.save(model_path)
    print(f">>> Training finished. Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()
