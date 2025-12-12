import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.algorithms.qlearning_ai import GomokuEnv
from backend.algorithms.classic_ai import GreedyAgent
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

def train_dqn_v2():
    # Configuration
    BOARD_SIZE = 15
    MODEL_PATH = "models/dqn_gomoku_v2" 
    TOTAL_TIMESTEPS = 1_000_000 # Can be increased for better results
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Target Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Model Path: {MODEL_PATH}")

    # Environment setup
    # Explicitly using GreedyAgent as training opponent
    opponent = GreedyAgent(distance=2)
    env = GomokuEnv(
        board_size=BOARD_SIZE,
        opponent_ai=opponent,
        reward_type='heuristic',
        invalid_penalty=-200.0
    )

    # Network Architecture
    # 512x512x256 is a reasonable size for 15x15 board features
    policy_kwargs = dict(net_arch=[512, 512, 256])

    # Load existing model or create new one
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f">>> Found existing model: {MODEL_PATH}.zip")
        print(">>> Loading model to CONTINUE training...")
        try:
            model = DQN.load(MODEL_PATH, env=env, device=device)
            reset_timesteps = False
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting FRESH training due to load error...")
            model = None
    else:
        model = None

    if model is None:
        print(">>> No existing model found (or load failed). Starting FRESH training...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            batch_size=512,
            learning_rate=1e-4,
            buffer_size=100_000,
            exploration_fraction=0.2, # Exploration reduces over first 20% of training
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log="./data/logs/dqn_v2_tensorboard/",
            policy_kwargs=policy_kwargs
        )
        reset_timesteps = True

    # Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path='./data/models/checkpoints_v2/',
        name_prefix='dqn_v2'
    )

    print("Training started... Press Query+C to stop safely.")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="run_v2_greedy_opponent",
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.zip")
        return

    model.save(MODEL_PATH)
    print(f"Training completed. Model saved to {MODEL_PATH}.zip")

if __name__ == "__main__":
    train_dqn_v2()
