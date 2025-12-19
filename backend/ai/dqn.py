"""DQN Agent and Gym Environment for Gomoku."""
import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any, List
from stable_baselines3 import DQN
from backend.engine.game_engine import GameEngine
from backend.engine.board import Board
from backend.ai.baselines import GreedyAgent

class GomokuEnv(gym.Env):
    """Custom OpenAI Gym env for Gomoku."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, opponent_list=None, reward_type='heuristic', invalid_penalty=-50.0):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size)
        
        self.opponents = opponent_list if opponent_list else [GreedyAgent()]
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
        self.current_opponent = random.choice(self.opponents)
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = divmod(int(action), self.board_size)

        if not self.engine.board.is_valid_move(x, y):
            return self._get_obs(), self.invalid_penalty, True, False, {"error": "Invalid"}

        self.engine.make_move(x, y) # AI Moves

        current_reward = 0.0
        done = False

        # AI Win Check
        if self.engine.game_over:
            if self.engine.winner == 1:
                current_reward = 200.0
                done = True
            else:
                current_reward = 0.0
                done = True
        else:
            # Heuristic Reward
            if self.reward_type == 'heuristic':
                current_reward += self._calculate_heuristic_reward(x, y, player=1)
                if self.engine.board.move_count < 10:
                    mid = self.board_size // 2
                    if abs(x - mid) + abs(y - mid) < 4:
                        current_reward += 0.5

        if done:
            return self._get_obs(), current_reward, done, False, {}

        # Opponent Move
        opp_x, opp_y = self.current_opponent.get_move(self.engine.board, 2)
        if opp_x != -1:
            self.engine.make_move(opp_x, opp_y)

        # Opponent Win Check
        if self.engine.game_over:
            if self.engine.winner == 2:
                return self._get_obs(), -200.0, True, False, {}
            return self._get_obs(), 0.0, True, False, {}

        return self._get_obs(), current_reward, False, False, {}

    def _calculate_heuristic_reward(self, x, y, player):
        """Calculate local heuristic reward for move (x,y)."""
        score = 0.0
        board = self.engine.board
        opponent = 3 - player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        # Offense
        for dx, dy in directions:
            count = 1
            tx, ty = x + dx, y + dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == player:
                count += 1; tx += dx; ty += dy
            tx, ty = x - dx, y - dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == player:
                count += 1; tx -= dx; ty -= dy
            
            if count == 5: score += 50.0
            elif count == 4: score += 15.0
            elif count == 3: score += 5.0
            elif count == 2: score += 1.0

        # Defense
        board.board[x][y] = opponent
        for dx, dy in directions:
            count = 1
            tx, ty = x + dx, y + dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == opponent:
                count += 1; tx += dx; ty += dy
            tx, ty = x - dx, y - dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == opponent:
                count += 1; tx -= dx; ty -= dy
            
            if count >= 5: score += 50.0
            elif count == 4: score += 15.0
            elif count == 3: score += 5.0
        
        board.board[x][y] = player
        return score

    def _get_obs(self):
        return np.array(self.engine.board.board, dtype=np.float32)

class QLearningAgent:
    """Agent wrapped around Stable Baselines3 DQN model."""
    
    def __init__(self, model_path: str = "models/dqn_gomoku", train_mode: bool = False):
        self.model_path = model_path
        self.model: Optional[DQN] = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path + ".zip"):
            try:
                self.model = DQN.load(self.model_path, device="cpu")
            except Exception as e:
                print(f"Warning: Failed to load DQN: {e}")

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        from backend.ai.baselines import random_move
        
        if self.model is None:
            return random_move(board)
            
        try:
            obs = np.array(board.board, dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            
            if isinstance(action, np.ndarray):
                action = int(action)
                
            x, y = divmod(int(action), 15)
            
            if not board.is_valid_move(x, y):
                return random_move(board)
                
            return (x, y)
            
        except Exception as e:
            print(f"DQN Error: {e}")
            return random_move(board)