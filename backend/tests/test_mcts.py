import unittest
import os
import shutil
import numpy as np
from backend.engine.board import Board
from backend.engine.game_engine import GameEngine
from backend.ai.mcts import MCTSAgent, get_neighbor_moves
from backend.ai.dqn import QLearningAgent, GomokuEnv


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
                shutil.rmtree("tests")
            except:
                pass

    def test_neighbor_moves(self):
        """Neighbors Check"""
        self.board.place_stone(7, 7, 1)
        moves = get_neighbor_moves(self.board, distance=1)
        
        self.assertEqual(len(moves), 8)
        self.assertIn((6, 6), moves)
        self.assertIn((8, 8), moves)
        
    def test_mcts_response(self):
        """MCTS Valid Move"""
        agent = MCTSAgent(time_limit=100, iteration_limit=10)
        
        self.board.place_stone(7, 7, 1)
        self.board.place_stone(7, 8, 2)
        
        move = agent.get_move(self.board, player=1)
        
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        self.assertTrue(self.board.is_valid_move(move[0], move[1]))

    def test_dqn_env_step(self):
        """DQN Environment Step"""
        env = GomokuEnv(board_size=15)
        obs, _ = env.reset()
        
        self.assertEqual(obs.shape, (15, 15))
        
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        
        self.assertEqual(env.engine.board.board[0][0], 1)
        self.assertFalse(done)
        self.assertIsInstance(reward, float)

    def test_dqn_agent_loading(self):
        """DQN Agent load fail fallback"""
        # Suppress print
        agent = QLearningAgent(model_path="path/to/nothing")
        
        move = agent.get_move(self.board, player=1)
        
        self.assertIsInstance(move, tuple)
        self.assertTrue(self.board.is_valid_move(move[0], move[1]))


if __name__ == '__main__':
    unittest.main()
