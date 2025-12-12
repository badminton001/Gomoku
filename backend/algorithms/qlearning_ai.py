import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Any
from stable_baselines3 import DQN
from backend.models.game_engine import GameEngine
from backend.models.board import Board
from backend.algorithms.classic_ai import GreedyAgent

class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=15, opponent_ai=None, reward_type='heuristic', invalid_penalty=-50.0):
        super().__init__()
        self.board_size = board_size
        self.engine = GameEngine(size=board_size)
        self.opponent_ai = opponent_ai if opponent_ai is not None else GreedyAgent()
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

        self.engine.make_move(x, y) # AI (Player 1) moves

        current_reward = 0.0
        done = False

        if self.engine.game_over:
            if self.engine.winner == 1:
                current_reward = 100.0
                done = True
            else: # Should not happen if player 1 just moved and game over
                current_reward = 0.0 # Draw or weird state
                done = True
        else:
            if self.reward_type == 'heuristic':
                current_reward += self._calculate_heuristic_reward(x, y, player=1)
                # Small position center bonus
                if self.engine.board.move_count < 10:
                    center = self.board_size // 2
                    if abs(x - center) + abs(y - center) < 4:
                        current_reward += 0.5
            else:
                current_reward = 0.0

        if done:
            return self._get_obs(), current_reward, done, False, {}

        # Opponent (Player 2) moves
        opp_x, opp_y = self.opponent_ai.get_move(self.engine.board, 2)
        if opp_x != -1:
            self.engine.make_move(opp_x, opp_y)

        if self.engine.game_over:
            if self.engine.winner == 2:
                return self._get_obs(), -100.0, True, False, {}
            return self._get_obs(), 0.0, True, False, {} # Draw

        return self._get_obs(), current_reward, False, False, {}

    def _calculate_heuristic_reward(self, x, y, player):
        score = 0.0
        board = self.engine.board
        opponent = 3 - player
        
        # 1. Offensive Rewards (Forming chains)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            # Count own length
            # Temporarily assume we are player (we are, but checking logic)
            # board already has player at x,y
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
            
            # Simple density rewards
            if count == 5:
                score += 50.0  # Will be capped by game over usually, but good to have
            elif count == 4:
                score += 15.0
            elif count == 3:
                score += 5.0
            elif count == 2:
                score += 1.0

        # 2. Defensive Rewards (Blocking)
        # Check what the opponent would have had if they played here
        board.board[x][y] = opponent # Swap query
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
            
            # If we blocked a 4 (which would have been 5), HUGE reward
            if count >= 5:
                score += 50.0 # Saved the game
            elif count == 4:
                score += 15.0 # Blocked a 4
            elif count == 3:
                score += 5.0  # Blocked a 3
        
        board.board[x][y] = player # Restore
        
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
            # Fallback to random if no model
            from backend.algorithms.classic_ai import random_move
            return random_move(board)
            
        obs = np.array(board.board, dtype=np.float32)
        # Predict
        action, _ = self.model.predict(obs, deterministic=True)
        x, y = divmod(int(action), 15)
        
        # Validation
        if not board.is_valid_move(x, y):
            # Fallback
            from backend.algorithms.classic_ai import random_move
            return random_move(board)
            
        return (x, y)

# Simple test block to verify environment
if __name__ == "__main__":
    env = GomokuEnv()
    obs, info = env.reset()
    print("Environment reset successful.")
    
    # Simulate a few steps
    # Center move
    action = 7 * 15 + 7
    obs, reward, done, truncated, info = env.step(action)
    print(f"step(center) -> reward={reward}, done={done}")
    
    if not done:
        # Check opponent move effect
        board_sum = np.sum(obs) # Should be at least 2 stones (1 ours, 1 opponent)
        print(f"Board sum (stones): {board_sum}")