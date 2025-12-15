import sys
import os
import unittest
import shutil
from datetime import datetime

# Path Fix (Prevent ModuleNotFoundError)
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.models.replay import GameReplay, Move
from backend.services.replay_service import ReplayService
from backend.services.move_scorer import MoveScorer


class TestPersonBServices(unittest.TestCase):

    def setUp(self):
        """
        Setup before each test.
        """
        # Set temp directory
        self.test_base_dir = os.path.join(current_dir, "temp_test_output")
        self.test_games_dir = os.path.join(self.test_base_dir, "games")

        # Initialize services
        self.replay_service = ReplayService(data_dir=self.test_games_dir)
        self.scorer = MoveScorer()

        # Force scorer output to temp dir
        self.scorer.charts_dir = os.path.join(self.test_base_dir, "charts")
        self.scorer.stats_dir = os.path.join(self.test_base_dir, "stats")

        # Create directories
        for d in [self.test_games_dir, self.scorer.charts_dir, self.scorer.stats_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def tearDown(self):
        """
        Cleanup after tests.
        """
        pass

    def test_full_workflow_pandas_and_matplotlib(self):
        """
        Core Test: Validate Pandas data organization + Matplotlib plotting.
        """
        print("\nTesting: Full Scoring Workflow (Table & Chart Generation)...")

        # 1. Prepare Fake Data
        moves = [
            Move(step=1, player=1, x=7, y=7, timestamp=100.1),
            Move(step=2, player=2, x=7, y=8, timestamp=100.5),
            Move(step=3, player=1, x=7, y=6, timestamp=101.0),
            Move(step=4, player=2, x=0, y=0, timestamp=102.0),
            Move(step=5, player=1, x=7, y=5, timestamp=103.0)
        ]

        game_id = "test_visualization_001"

        # 2. Call Scorer
        result = self.scorer.score_moves(moves, game_id=game_id)

        # 3. Verify Result Keys
        required_keys = ["score_curve", "critical_moments", "chart_path", "stats_summary", "csv_path"]
        for key in required_keys:
            self.assertIn(key, result, f"Result missing key: {key}")

        # 4. Verify Stats
        stats = result["stats_summary"]
        print(f"Stats Summary: {stats}")
        self.assertIn("mean_score", stats)
        self.assertIn("brilliant_count", stats)
        self.assertIsInstance(stats["mean_score"], float)

        # 5. Verify File Generation
        # Verify CSV
        csv_path = result["csv_path"]
        self.assertTrue(os.path.exists(csv_path), f"CSV file not generated: {csv_path}")
        print(f"CSV Generated: {csv_path}")

        # Verify PNG
        chart_path = result["chart_path"]
        self.assertTrue(os.path.exists(chart_path), f"Chart file not generated: {chart_path}")
        print(f"PNG Chart Generated: {chart_path}")

    def test_replay_save_load(self):
        """
        Basic Test: Verify Replay Service Save/Load.
        """
        print("\nTesting: Replay Service Save/Load...")
        fake_game = GameReplay(
            game_id="test_io_001",
            start_time=datetime.now(),
            winner=1,
            moves=[Move(step=1, player=1, x=7, y=7, timestamp=100.0)]
        )

        # Save
        path = self.replay_service.save_replay(fake_game)
        self.assertTrue(os.path.exists(path))

        # Load
        loaded = self.replay_service.load_replay("test_io_001")
        self.assertEqual(loaded['game_id'], "test_io_001")
        print("Save/Load Functional")


if __name__ == '__main__':
    unittest.main()