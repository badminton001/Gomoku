import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from backend.models.game_engine import GameEngine
from backend.models.board import Board


class GomokuEnv(gym.Env):
    """五子棋 Gymnasium 环境"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, opponent_ai=None):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size)
        self.opponent_ai = opponent_ai
        
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(board_size * board_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset_game()
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = divmod(int(action), self.board_size)
        
        if not self.engine.board.is_valid_move(x, y):
            return self._get_obs(), -100.0, True, False, {"error": "Invalid move"}

        self.engine.make_move(x, y)

        if self.engine.game_over:
            if self.engine.winner == 1:
                return self._get_obs(), 100.0, True, False, {}
            else:
                return self._get_obs(), 0.0, True, False, {}

        opp_x, opp_y = -1, -1
        if self.opponent_ai:
            try:
                move = self.opponent_ai.get_move(self.engine.board, self.engine.current_player)
                opp_x, opp_y = move
            except:
                opp_x, opp_y = self._random_empty_move()
        else:
            opp_x, opp_y = self._random_empty_move()
            
        if opp_x != -1:
            self.engine.make_move(opp_x, opp_y)

        if self.engine.game_over:
            if self.engine.winner == 2:
                return self._get_obs(), -100.0, True, False, {}
            else:
                return self._get_obs(), 0.0, True, False, {}

        return self._get_obs(), -0.1, False, False, {}

    def _get_obs(self):
        return np.array(self.engine.board.board, dtype=np.float32)

    def _random_empty_move(self):
        import random
        empty = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.engine.board.board[i][j] == 0:
                    empty.append((i, j))
        if not empty:
            return -1, -1
        return random.choice(empty)

    def render(self):
        self.engine.debug_print_board()


class QLearningAgent:
    """DQN 代理"""
    
    def __init__(self, model_path: str = "models/dqn_gomoku", train_mode: bool = False):
        self.model_path = model_path
        self.model: Optional[DQN] = None
        self.load_model()
        
        if self.model is None and not train_mode:
            print(f"[DQN] No model at {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path + ".zip"):
            try:
                self.model = DQN.load(self.model_path)
                print(f"[DQN] Loaded {self.model_path}")
            except Exception as e:
                print(f"[DQN] Load failed: {e}")

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        """获取着法"""
        if self.model is None:
            from backend.algorithms.classic_ai import random_move
            return random_move(board)

        obs = np.array(board.board, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        
        x, y = divmod(int(action), 15)
        
        if not board.is_valid_move(x, y):
            from backend.algorithms.classic_ai import random_move
            return random_move(board)
            
        return (x, y)

    def evaluate_board(self, board: Board, player: int) -> float:
        """评估棋盘状态 (0-100)"""
        if self.model is None:
            return 50.0

        obs = np.array(board.board, dtype=np.float32)
        import torch
        with torch.no_grad():
            obs_tensor = self.model.policy.obs_to_tensor(obs)
            q_values = self.model.q_net(obs_tensor)
            max_q = torch.max(q_values).item()
            
        score = 50 + (max_q / 2.0)
        return max(0.0, min(100.0, score))
