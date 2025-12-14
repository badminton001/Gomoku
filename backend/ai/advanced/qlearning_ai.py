import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any, List
from stable_baselines3 import DQN
from backend.models.game_engine import GameEngine
from backend.models.board import Board
from backend.algorithms.classic_ai import GreedyAgent, RandomAgent

class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, opponent_list=None, reward_type='heuristic', invalid_penalty=-50.0):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size)
        
        # Initialize opponents list
        if opponent_list is None:
             # Default to GreedyAgent if none provided
             self.opponents = [GreedyAgent()]
        else:
             self.opponents = opponent_list
        
        # Pick one for starters
        self.current_opponent = self.opponents[0]

        self.reward_type = reward_type
        self.invalid_penalty = invalid_penalty
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(board_size * board_size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset_game()
        
        # Randomly select an opponent for this new episode
        self.current_opponent = random.choice(self.opponents)
        
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = divmod(int(action), self.board_size)

        if not self.engine.board.is_valid_move(x, y):
            return self._get_obs(), self.invalid_penalty, True, False, {"error": "Invalid"}

        self.engine.make_move(x, y) # AI (Player 1) moves

        current_reward = 0.0
        done = False

        if self.engine.game_over:
            if self.engine.winner == 1:
                current_reward = 200.0 # Increased from 100 for winning
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

        # Opponent (Player 2) moves
        opp_x, opp_y = self.current_opponent.get_move(self.engine.board, 2)
        if opp_x != -1:
            self.engine.make_move(opp_x, opp_y)

        if self.engine.game_over:
            if self.engine.winner == 2:
                return self._get_obs(), -200.0, True, False, {} # Increased penalty
            return self._get_obs(), 0.0, True, False, {}

        return self._get_obs(), current_reward, False, False, {}

    def _calculate_heuristic_reward(self, x, y, player):
        score = 0.0
        board = self.engine.board
        opponent = 3 - player
        
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
            
            if count == 5:
                score += 50.0
            elif count == 4:
                score += 15.0
            elif count == 3:
                score += 5.0
            elif count == 2:
                score += 1.0

        # Blocking rewards
        board.board[x][y] = opponent
        for dx, dy in directions:
            count = 1
            tx, ty = x + dx, y + dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == opponent:
                count += 1
                tx += dx
                ty += dy
            tx, ty = x - dx, y - dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == opponent:
                count += 1
                tx -= dx
                ty -= dy
            
            if count >= 5:
                score += 50.0
            elif count == 4:
                score += 15.0
            elif count == 3:
                score += 5.0
        
        board.board[x][y] = player
        return score

    def _get_obs(self):
        return np.array(self.engine.board.board, dtype=np.float32)

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