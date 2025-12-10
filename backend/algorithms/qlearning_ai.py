import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any
from stable_baselines3 import DQN
from backend.models.game_engine import GameEngine
from backend.models.board import Board


class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, opponent_ai=None, reward_type='heuristic', invalid_penalty=-50.0):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size)
        self.opponent_ai = opponent_ai
        self.reward_type = reward_type
        self.invalid_penalty = invalid_penalty
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
            return self._get_obs(), self.invalid_penalty, True, False, {"error": "Invalid"}

        self.engine.make_move(x, y)

        current_reward = 0.0
        done = False

        if self.engine.game_over:
            if self.engine.winner == 1:
                current_reward = 100.0
                done = True
            else:
                current_reward = 0.0
                done = True
        else:
            if self.reward_type == 'heuristic':
                current_reward += self._calculate_heuristic_reward(x, y, player=1)

                if self.engine.board.move_count < 10:
                    center = self.board_size // 2
                    if abs(x - center) + abs(y - center) < 4:
                        current_reward += 0.5
            else:
                current_reward = 0.0

        if done:
            return self._get_obs(), current_reward, done, False, {}

        opp_x, opp_y = self._random_empty_move()
        if opp_x != -1:
            self.engine.make_move(opp_x, opp_y)

        if self.engine.game_over:
            if self.engine.winner == 2:
                return self._get_obs(), -100.0, True, False, {}
            return self._get_obs(), 0.0, True, False, {}

        return self._get_obs(), current_reward, False, False, {}

    def _calculate_heuristic_reward(self, x, y, player):
        score = 0.0
        board = self.engine.board
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            tx, ty = x + dx, y + dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == player:
                count += 1
                tx += dx
                ty += dy
            tx, ty = x - dx, y - dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == player:
                count += 1
                tx -= dx
                ty -= dy

            if count == 4:
                score += 10.0
            elif count == 3:
                score += 2.0
            elif count == 2:
                score += 0.5
        return score

    def _get_obs(self):
        return np.array(self.engine.board.board, dtype=np.float32)

    def _random_empty_move(self):
        import random
        empty = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.engine.board.board[i][j] == 0:
                    empty.append((i, j))
        if not empty: return -1, -1
        return random.choice(empty)


class QLearningAgent:
    def __init__(self, model_path: str = "models/dqn_gomoku", train_mode: bool = False):
        self.model_path = model_path
        self.model: Optional[DQN] = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path + ".zip"):
            try:
                self.model = DQN.load(self.model_path)
            except Exception as e:
                pass

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
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