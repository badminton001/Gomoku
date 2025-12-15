import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai.advanced.qlearning_ai import GomokuEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def train_15x15_final():
    BOARD_SIZE = 15
    MODEL_PATH = "models/dqn_15x15_final"
    TOTAL_TIMESTEPS = 5_000_000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Target Timesteps: {TOTAL_TIMESTEPS}")

    env = GomokuEnv(
        board_size=BOARD_SIZE,
        reward_type='heuristic',
        invalid_penalty=-200.0
    )

    policy_kwargs = dict(net_arch=[512, 512, 256])

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f">>> Found existing model: {MODEL_PATH}.zip")
        print(">>> Loading model to CONTINUE training...")

        model = DQN.load(MODEL_PATH, env=env, device=device)
        reset_timesteps = False
    else:
        print(">>> No existing model found. Starting FRESH training...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            batch_size=512,
            learning_rate=1e-4,
            buffer_size=200_000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tensorboard_log="./data/logs/dqn_15x15_final_tensorboard/",
            policy_kwargs=policy_kwargs
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path='./data/models/checkpoints_15x15_final/',
        name_prefix='dqn_15x15'
    )

    print("Training started...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="run_15x15_strict",
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.zip")
        return

    model.save(MODEL_PATH)
    print(f"Training completed. Model saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    train_15x15_final()