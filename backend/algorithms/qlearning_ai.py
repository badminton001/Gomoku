import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
from backend.models.game_engine import GameEngine
from backend.algorithms.mcts_ai import get_neighbor_moves
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os


class GomokuEnv(gym.Env):
    """五子棋环境"""
    metadata = {"render_modes": ["human"]}
    def __init__(self, opponent_ai=None):
        super().__init__()
        self.engine = GameEngine(size=15, first_player=1)
        self.opponent_ai = opponent_ai
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(15, 15), dtype=np.int32
        )
        self.action_space = spaces.Discrete(225)

    def _board_to_obs(self) -> np.ndarray:
        """棋盘转观察"""
        return np.array(self.engine.board.board, dtype=np.int32)

    def _action_to_pos(self, action: int) -> Tuple[int, int]:
        """动作转坐标"""
        return divmod(action, 15)

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        self.engine.reset_game()
        return self._board_to_obs(), {}

    def step(self, action: int):
        """执行动作"""
        x, y = self._action_to_pos(action)
        # 玩家下棋
        success = self.engine.make_move(x, y)
        if not success:
            return self._board_to_obs(), -1.0, False, False, {}
        # 检查是否结束
        if self.engine.game_over:
            if self.engine.winner == 1:
                return self._board_to_obs(), 1.0, True, False, {}
            elif self.engine.winner == 2:
                return self._board_to_obs(), -1.0, True, False, {}
            else:
                return self._board_to_obs(), 0.0, True, False, {}
        # 对手下棋
        if self.opponent_ai:
            opp_move = self.opponent_ai.get_move(self.engine.board, self.engine.current_player)
            success = self.engine.make_move_for(opp_move, opp_move, self.engine.current_player)
            if not success:
                return self._board_to_obs(), -1.0, False, False, {}
            if self.engine.game_over:
                if self.engine.winner == 1:
                    return self._board_to_obs(), 1.0, True, False, {}
                elif self.engine.winner == 2:
                    return self._board_to_obs(), -1.0, True, False, {}
                else:
                    return self._board_to_obs(), 0.0, True, False, {}

        return self._board_to_obs(), 0.0, False, False, {}

    def render(self):
        self.engine.debug_print_board()


class SaveBestModelCallback(BaseCallback):
    """保存最优模型"""
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.model.num_timesteps % 5000 == 0:
            current_reward = np.mean(self.model.ep_info_buffer.rewards)
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.model.save(self.save_path)
        return True


class QLearningAgent:
    """Q-Learning / DQN 代理"""
    def __init__(self, model_path: str = "models/dqn_gomoku"):
        self.model_path = model_path
        self.model = None
        self.env = None

    def train(self, total_timesteps: int = 100000, opponent_ai=None):
        """训练模型"""
        self.env = GomokuEnv(opponent_ai=opponent_ai)

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=1,
            gradient_steps=-1,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1
        )
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        callback = SaveBestModelCallback(save_path=self.model_path)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=100
        )
        self.model.save(self.model_path)

    def load(self):
        """加载模型"""
        self.env = GomokuEnv()
        self.model = DQN.load(self.model_path, env=self.env)

    def get_move(self, board, player: int) -> Tuple[int, int]:
        """获取着法"""
        if self.model is None:
            self.load()
        obs = np.array(board.board, dtype=np.int32)
        action, _ = self.model.predict(obs, deterministic=True)
        x, y = divmod(int(action), 15)
        if board.is_valid_move(x, y):
            return (x, y)
        moves = get_neighbor_moves(board)
        return moves if moves else (7, 7)

    def evaluate_board(self, board, player: int = 1) -> float:
        """评估棋盘"""
        if self.model is None:
            self.load()
        obs = np.array(board.board, dtype=np.int32)
        q_values = self.model.q_net(self.model._get_obs_from_array(obs)).detach().numpy()
        max_q = np.max(q_values)
        score = (max_q + 1) * 50
        return min(100, max(0, float(score)))