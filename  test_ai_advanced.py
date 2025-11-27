import unittest
import os
from board import Board
from ai_advanced import MCTSAgent, QLearningAgent, get_neighbor_moves


class TestAI(unittest.TestCase):

    def setUp(self):
        self.board = Board(size=15)

    def test_neighbor_moves_logic(self):
        """空盘返回中心，落子后返回周围空位"""
        # 空盘
        moves = get_neighbor_moves(self.board)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0], (7, 7))

        # 下一颗子在 (7,7)
        self.board.place_stone(7, 7, 1)
        moves = get_neighbor_moves(self.board, distance=1)
        # 3x3 减去中间的一个 = 8
        self.assertEqual(len(moves), 8)
        self.assertIn((6, 6), moves)
        self.assertIn((8, 8), moves)
        print("[Test] Neighbor moves logic passed.")

    def test_mcts_basic(self):
        """MCTS 返回合法移动"""
        agent = MCTSAgent(time_limit=0.5, max_iterations=50)

        # 模拟局面
        self.board.place_stone(7, 7, 1)
        self.board.place_stone(7, 8, 2)

        move = agent.get_move(self.board, player=1)

        self.assertTrue(self.board.is_valid_move(move[0], move[1]))
        print(f"[Test] MCTS returned move: {move}")

    def test_q_learning_save_load(self):
        """Q表保存与加载"""
        agent = QLearningAgent()
        agent.set_q("fake_state", (0, 0), 10.5)

        filename = "test_q_model.pkl"
        agent.save_model(filename)

        # 创建新Agent加载
        new_agent = QLearningAgent()
        new_agent.load_model(filename)

        val = new_agent.get_q("fake_state", (0, 0))
        self.assertEqual(val, 10.5)

        # 清理
        if os.path.exists(filename):
            os.remove(filename)
        print("[Test] Q-Learning Save/Load passed.")

    def test_q_learning_learn(self):
        """Q值更新逻辑验证"""
        agent = QLearningAgent(alpha=0.5, gamma=0.9)

        # 设置上一步状态
        agent.last_state = "state1"
        agent.last_action = (1, 1)

        # 初始 Q 值为 0
        self.assertEqual(agent.get_q("state1", (1, 1)), 0.0)

        # 学习：给予奖励 10
        agent.learn(self.board, reward=10.0)

        # 新 Q 值 = 0 + 0.5 * (10 + 0 - 0) = 5.0
        self.assertEqual(agent.get_q("state1", (1, 1)), 5.0)
        print("[Test] Q-Learning update logic passed.")


if __name__ == '__main__':
    unittest.main()