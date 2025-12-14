import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.algorithms.qlearning_ai import GomokuEnv
from backend.algorithms.classic_ai import GreedyAgent, RandomAgent
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def train_dqn_v3():
    # Configuration for Version 3
    BOARD_SIZE = 15
    MODEL_PATH = "models/dqn_gomoku_v3"
    TOTAL_TIMESTEPS = 5_000_000 
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Target Timesteps: {TOTAL_TIMESTEPS}")

    # Mixed Opponents: 
    # 1. RandomAgent: Prevents overfitting to deterministic strategies
    # 2. Greedy(dist=1): Short-sighted aggressive
    # 3. Greedy(dist=2): Standard aggressive
    opponents = [
        RandomAgent(), 
        GreedyAgent(distance=1), 
        GreedyAgent(distance=2)
    ]
    
    env = GomokuEnv(
        board_size=BOARD_SIZE,
        opponent_list=opponents, 
        reward_type='heuristic',
        invalid_penalty=-200.0
    )

    policy_kwargs = dict(net_arch=[512, 512, 256])

    # Resume from V3 checkpoint if exists, otherwise fresh
    # Can also load V2 as starting point if desired, but fresh V3 is cleaner for mixed opponent learning
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = DQN.load(MODEL_PATH, env=env, device=device)
    else:
        print("Creating new DQN V3 model...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            batch_size=512,
            learning_rate=1e-4,
            buffer_size=100_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=policy_kwargs
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=250_000, 
        save_path='./data/models/checkpoints_v3/',
        name_prefix='dqn_v3'
    )

    print("Training started on Kaggle GPU with Mixed Opponents (V3)...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="run_v3_mixed"
    )

    model.save(MODEL_PATH)
    print(f"Training completed. Model saved to {MODEL_PATH}.zip")

if __name__ == "__main__":
    train_dqn_v3()
