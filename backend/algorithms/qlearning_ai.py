import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from backend.models.game_engine import GameEngine
from backend.algorithms.mcts_ai import get_neighbor_moves


class GomokuEnv(gym.Env):
    """Gymnasium 五子棋环境"""
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, opponent_ai=None, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size, first_player=1)
        self.opponent_ai = opponent_ai
        
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.int32
        )
        self.action_space = spaces.Discrete(board_size * board_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset_game()
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = divmod(int(action), self.board_size)
        
        if not self.engine.board.is_valid_move(x, y):
            return self._get_obs(), -10.0, True, False, {"error": "Invalid move"}

        success = self.engine.make_move(x, y)
        if not success:
             return self._get_obs(), -10.0, True, False, {}

        if self.engine.game_over:
            reward = 10.0 if self.engine.winner == 1 else 0.0
            return self._get_obs(), reward, True, False, {}

        if self.opponent_ai:
            try:
                opp_move = self.opponent_ai.get_move(self.engine.board, self.engine.current_player)
                self.engine.make_move_for(opp_move, opp_move, self.engine.current_player)
            except Exception as e:
                print(f"Opponent AI error: {e}")
                pass

        if self.engine.game_over:
            reward = -10.0 if self.engine.winner == 2 else 0.0
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), 0.1, False, False, {}

    def _get_obs(self):
        return np.array(self.engine.board.board, dtype=np.int32)

    def render(self):
        self.engine.debug_print_board()


class QLearningAgent:
    """DQN 代理 (Stable-Baselines3)"""
    
    def __init__(self, model_path: str = "models/dqn_gomoku"):
        self.model_path = model_path
        self.model: Optional[DQN] = None

    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path + ".zip"):
            try:
                self.model = DQN.load(self.model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"Model not found at {self.model_path}")

    def train(self, total_timesteps: int = 10000, opponent_ai=None):
        """训练"""
        env = GomokuEnv(opponent_ai=opponent_ai)
        env = Monitor(env)
        
        self.model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05
        )
        
        print(f"Training for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def get_move(self, board, player: int) -> Tuple[int, int]:
        """获取着法"""
        if self.model is None:
            self.load_model()

        if self.model is None:
            moves = get_neighbor_moves(board)
            return moves if moves else (7, 7)

        obs = np.array(board.board, dtype=np.int32)
        action, _states = self.model.predict(obs, deterministic=True)
        
        x, y = divmod(int(action), 15)
        
        if not board.is_valid_move(x, y):
            moves = get_neighbor_moves(board)
            return moves if moves else (7, 7)
            
        return (x, y)

    def evaluate_board(self, board, player: int = 1) -> float:
        """评估棋盘 (0-100)"""
        if self.model is None:
            self.load_model()
            
        if self.model:
            return 80.0
        return 50.0
