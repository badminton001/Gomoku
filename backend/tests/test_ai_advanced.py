import unittest
import os
import shutil
import numpy as np
from backend.models.board import Board
from backend.models.game_engine import GameEngine
from backend.algorithms.mcts_ai import MCTSAgent, get_neighbor_moves
from backend.algorithms.qlearning_ai import QLearningAgent, GomokuEnv


class TestAdvancedAI(unittest.TestCase):

    def setUp(self):
        self.board = Board(size=15)
        self.engine = GameEngine(size=15)
        self.test_model_path = "tests/temp_dqn_model"

    def tearDown(self):
        if os.path.exists(self.test_model_path + ".zip"):
            os.remove(self.test_model_path + ".zip")
        if os.path.exists("tests"):
            try:
                os.rmdir("tests")
            except:
                pass

    def test_neighbor_moves(self):
        """邻域生成"""
        moves = get_neighbor_moves(self.board)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves, (7, 7))

        self.board.place_stone(7, 7, 1)
        moves = get_neighbor_moves(self.board, distance=1)
        self.assertEqual(len(moves), 8)
        self.assertIn((6, 6), moves)
        
    def test_mcts_response(self):
        """MCTS 着法返回"""
        agent = MCTSAgent(time_limit=0.1, iteration_limit=10)
        
        self.board.place_stone(7, 7, 1)
        self.board.place_stone(7, 8, 2)
        
        move = agent.get_move(self.board, player=1)
        
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        self.assertTrue(self.board.is_valid_move(move, move))
        print(f"MCTS move: {move}")

    def test_dqn_env_step(self):
        """环境 step 测试"""
        env = GomokuEnv(opponent_ai=None, board_size=15)
        obs, _ = env.reset()
        
        self.assertEqual(obs.shape, (15, 15))
        
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        
        self.assertEqual(env.engine.board.board, 1)
        self.assertFalse(done)
        print("Env step test passed")

    def test_dqn_agent_train_and_predict(self):
        """DQN 训练和预测"""
        agent = QLearningAgent(model_path=self.test_model_path)
        agent.train(total_timesteps=100)
        
        self.assertTrue(os.path.exists(self.test_model_path + ".zip"))
        
        new_agent = QLearningAgent(model_path=self.test_model_path)
        move = new_agent.get_move(self.board, player=1)
        
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        print(f"DQN prediction: {move}")


if __name__ == '__main__':
    unittest.main()
