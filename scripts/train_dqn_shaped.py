import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.algorithms.qlearning_ai import GomokuEnv
from stable_baselines3 import DQN

def train_shaped():
    BOARD_SIZE = 15
    MODEL_PATH = "models/benchmark_shaped"
    TOTAL_TIMESTEPS = 1_000_000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> [Experiment 2] Shaped Reward Training | Device: {device}")

    env = GomokuEnv(board_size=BOARD_SIZE, reward_type='heuristic', invalid_penalty=-50.0)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        batch_size=512,
        learning_rate=1e-4,
        tensorboard_log="./data/logs/benchmark_tensorboard/"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="2_Shaped_V1")
    model.save(MODEL_PATH)
    print(f">>> Model saved to {MODEL_PATH}.zip")

if __name__ == "__main__":
    train_shaped()