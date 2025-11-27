import unittest

from backend.models.board import Board
from backend.algorithms.classic_ai import (
    AlphaBetaAgent,
    GreedyAgent,
    MinimaxAgent,
    evaluate_board,
    random_move,
)


class TestClassicAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board(size=10)

    def test_random_move_is_valid(self):
        self.board.place_stone(5, 5, 1)
        move = random_move(self.board)
        self.assertTrue(self.board.is_valid_move(*move))

    def test_evaluate_board_prefers_winning_extension(self):
        for x in range(4):
            self.board.place_stone(x, 0, 1)
        base_score = evaluate_board(self.board, 1)
        self.board.place_stone(4, 0, 1)
        winning_score = evaluate_board(self.board, 1)
        self.assertGreater(winning_score, base_score)

    def test_greedy_finishes_five_in_row(self):
        for x in range(4):
            self.board.place_stone(x, 0, 1)
        agent = GreedyAgent()
        move = agent.get_move(self.board, 1)
        self.assertEqual(move, (4, 0))
        self.assertIsNotNone(agent.last_metrics)

    def test_minimax_blocks_immediate_threat(self):
        for x in range(4):
            self.board.place_stone(x, 0, 2)
        agent = MinimaxAgent(depth=2, candidate_limit=10)
        move = agent.get_move(self.board, 1)
        self.assertEqual(move, (4, 0))
        self.assertIsNotNone(agent.last_metrics)
        self.assertGreater(agent.last_metrics.explored_nodes, 0)

    def test_alphabeta_matches_minimax_choice(self):
        for x in range(4):
            self.board.place_stone(x, 0, 2)
        minimax_agent = MinimaxAgent(depth=2, candidate_limit=10)
        alphabeta_agent = AlphaBetaAgent(depth=2, candidate_limit=10)
        minimax_move = minimax_agent.get_move(self.board, 1)
        alphabeta_move = alphabeta_agent.get_move(self.board, 1)
        self.assertEqual(minimax_move, alphabeta_move)
        self.assertLessEqual(alphabeta_agent.last_metrics.explored_nodes, minimax_agent.last_metrics.explored_nodes)


if __name__ == "__main__":
    unittest.main()
