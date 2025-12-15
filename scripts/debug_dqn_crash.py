import sys
import os
import numpy as np
from stable_baselines3 import DQN

def debug_dqn():
    model_path = "models/dqn_15x15_final"
    print(f"Loading model from {model_path}...")
    
    try:
        model = DQN.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy observation (15x15)
    obs = np.zeros((15, 15), dtype=np.float32)
    print(f"Predicting with obs shape: {obs.shape}, dtype: {obs.dtype}")
    
    try:
        action, _states = model.predict(obs, deterministic=True)
        print(f"Prediction success! Action: {action}")
        print(f"Return type of action: {type(action)}")
    except Exception as e:
        print(f"Prediction Crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dqn()
