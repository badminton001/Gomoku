import json
import unittest
from pathlib import Path

from backend.ai.baselines import (
    GreedyAgent,
    RandomAgent,
    random_move,
)
from backend.ai.minimax import AlphaBetaAgent
from backend.engine.board import Board


class TestClassicAI(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board(size=10)

    def test_random_move_is_valid(self):
        self.board.place_stone(5, 5, 1)
        move = random_move(self.board)
        self.assertTrue(self.board.is_valid_move(*move))

    def test_greedy_finishes_five_in_row(self):
        for x in range(4):
            self.board.place_stone(x, 0, 1)
        agent = GreedyAgent()
        move = agent.get_move(self.board, 1)
        self.assertEqual(move, (4, 0))

    def test_minimax_blocks_immediate_threat(self):
        for x in range(4):
            self.board.place_stone(x, 0, 2)
        # Using AlphaBetaAgent as MinimaxAgent replacement
        agent = AlphaBetaAgent(depth=2, time_limit=2.0)
        move = agent.get_move(self.board, 1)
        self.assertEqual(move, (4, 0))

    def test_load_ai_config_json(self):
        path = Path("backend/config/ai_config.json")
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertIn("neighbor_distance", cfg)

if __name__ == "__main__":
    unittest.main()
