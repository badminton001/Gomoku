import sys
import os
import yaml
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.algorithms.qlearning_ai import GomokuEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def train():
    BOARD_SIZE = 15
    
    MODEL_PATH = f"models/dqn_gomoku_{BOARD_SIZE}x{BOARD_SIZE}"
    
    if BOARD_SIZE <= 10:
        TOTAL_TIMESTEPS = 200_000
        policy_kwargs = None
        batch_size = 256
        buffer_size = 50000
        learning_starts = 1000
    else:
        TOTAL_TIMESTEPS = 5_000_000
        policy_kwargs = dict(net_arch=[512, 512, 256])
        batch_size = 512
        buffer_size = 200000
        learning_starts = 50000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Mode: {BOARD_SIZE}x{BOARD_SIZE} | Device: {device.upper()}")

    env = GomokuEnv(board_size=BOARD_SIZE)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f">>> Loading existing model: {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env, device=device)
        reset_timesteps = False
    else:
        print(f">>> Creating NEW model for {BOARD_SIZE}x{BOARD_SIZE}...")
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_rate=1e-4,
            learning_starts=learning_starts,
            gamma=0.995 if BOARD_SIZE > 10 else 0.95,
            target_update_interval=2000 if BOARD_SIZE > 10 else 500,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            tensorboard_log=f"./data/logs/dqn_{BOARD_SIZE}x{BOARD_SIZE}_tensorboard/"
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f'./data/models/checkpoints_{BOARD_SIZE}x{BOARD_SIZE}/',
        name_prefix=f'dqn_{BOARD_SIZE}x{BOARD_SIZE}'
    )
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback, 
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        
    model.save(MODEL_PATH)
    print("Saved.")

if __name__ == "__main__":
    train()
