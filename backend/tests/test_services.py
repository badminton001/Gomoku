import sys
import os
import unittest
import shutil
from datetime import datetime


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
        """测试前的准备工作"""
        # 在 backend/services 目录下创建一个临时的 temp_data 文件夹用于测试
        # 为了避免路径混乱，使用绝对路径
        self.test_dir = os.path.join(current_dir, "temp_data_for_test")

        # 初始化服务
        self.replay_service = ReplayService(data_dir=self.test_dir)
        self.scorer = MoveScorer()

    def tearDown(self):
        """测试后的清理工作：删掉临时生成的测试文件"""
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                print(f"Warning: Could not delete temp dir {self.test_dir}")

    def test_save_and_load_replay(self):
        """测试：能不能把游戏保存成文件，再读出来"""
        print("\n正在测试: 回放保存与加载...")

        # 1. 造一个假的游戏记录
        fake_game = GameReplay(
            game_id="test_game_001",
            start_time=datetime.now(),
            winner=1,
            moves=[
                Move(step=1, player=1, x=7, y=7, timestamp=100.0),
                Move(step=2, player=2, x=7, y=8, timestamp=105.0)
            ]
        )

        # 2. 保存
        file_path = self.replay_service.save_replay(fake_game)

        # 验证文件是否存在
        self.assertTrue(os.path.exists(file_path), "保存失败：文件没生成")
        print(f"文件已保存至: {file_path}")

        # 3. 读取
        loaded_data = self.replay_service.load_replay("test_game_001")

        # 验证读取的数据
        self.assertEqual(loaded_data['game_id'], "test_game_001", "读取失败：ID对不上")
        self.assertEqual(len(loaded_data['moves']), 2, "读取失败：步数不对")
        print("pass 保存与加载测试通过！")

    def test_move_scoring(self):
        """测试：评分系统能否正常工作"""
        print("\n正在测试: AI着法评分...")

        moves = [
            Move(step=1, player=1, x=7, y=7, timestamp=100.0),
            Move(step=2, player=2, x=0, y=0, timestamp=105.0)
        ]

        # 调用评分器
        result = self.scorer.score_moves(moves)

        # 验证结果结构
        self.assertIn("score_curve", result)
        self.assertIn("critical_moments", result)
        self.assertEqual(len(result["score_curve"]), 2)
        print("✅ 评分系统接口测试通过！")


if __name__ == '__main__':
    unittest.main()