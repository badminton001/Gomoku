import json
import unittest
from pathlib import Path

from backend.algorithms.classic_ai import (
    AlphaBetaAgent,
    GreedyAgent,
    MinimaxAgent,
    SearchMetrics,
    evaluate_board,
    load_ai_config,
    random_move,
)
from backend.models.board import Board


class TestClassicAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board(size=10)

    def test_random_move_is_valid(self):
        self.board.place_stone(5, 5, 1)
        move = random_move(self.board)
        self.assertTrue(self.board.is_valid_move(*move))

    def test_evaluate_board_prefers_longer_runs(self):
        for x in range(3):
            self.board.place_stone(x, 0, 1)
        base_score = evaluate_board(self.board, 1)
        self.board.place_stone(3, 0, 1)
        longer_score = evaluate_board(self.board, 1)
        self.assertGreater(longer_score, base_score)

    def test_greedy_finishes_five_in_row(self):
        for x in range(4):
            self.board.place_stone(x, 0, 1)
        agent = GreedyAgent()
        move = agent.get_move(self.board, 1)
        self.assertEqual(move, (4, 0))
        self.assertIsInstance(agent.last_metrics, SearchMetrics)

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

    def test_load_ai_config_json(self):
        path = Path("backend/config/ai_config.json")
        cfg = load_ai_config(str(path))
        self.assertEqual(cfg["neighbor_distance"], 2)
        self.assertEqual(cfg["minimax"]["depth"], 2)
        # Ensure file is valid JSON
        json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
