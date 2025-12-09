import sys
import os
import unittest
import shutil
from datetime import datetime

# ================= 路径修复 (防止 ModuleNotFoundError) =================
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 backend 目录
backend_dir = os.path.dirname(current_dir)
# 获取 Gomoku 根目录
project_root = os.path.dirname(backend_dir)
# 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)
# ===================================================================

from backend.models.replay import GameReplay, Move
from backend.services.replay_service import ReplayService
from backend.services.move_scorer import MoveScorer


class TestPersonBServices(unittest.TestCase):

    def setUp(self):
        """
        每次测试前的准备工作
        """
        # 设置临时测试目录，避免污染真实数据
        self.test_base_dir = os.path.join(current_dir, "temp_test_output")
        self.test_games_dir = os.path.join(self.test_base_dir, "games")

        # 初始化服务 (注入测试路径)
        self.replay_service = ReplayService(data_dir=self.test_games_dir)
        self.scorer = MoveScorer()

        # 强制修改 Scorer 的输出目录到临时文件夹，方便测试后删除
        self.scorer.charts_dir = os.path.join(self.test_base_dir, "charts")
        self.scorer.stats_dir = os.path.join(self.test_base_dir, "stats")

        # 创建目录
        for d in [self.test_games_dir, self.scorer.charts_dir, self.scorer.stats_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def tearDown(self):
        """
        测试后的清理工作：自动删除生成的临时文件（此处已被注释，可以保存结果）
        """
        #if os.path.exists(self.test_base_dir):
         #   try:
          #      shutil.rmtree(self.test_base_dir)
          #      print(f"\n🧹 已清理临时测试文件: {self.test_base_dir}")
           # except Exception as e:
           #     print(f"清理失败: {e}")
        pass
    def test_full_workflow_pandas_and_matplotlib(self):
        """
        🔥 核心测试: 验证 Pandas 数据组织 + Matplotlib 画图功能
        """
        print("\n正在测试: 完整评分工作流 (含数据表和图表生成)...")

        # 1. 准备假数据 (模拟一局 5 步的棋)
        moves = [
            Move(step=1, player=1, x=7, y=7, timestamp=100.1),
            Move(step=2, player=2, x=7, y=8, timestamp=100.5),
            Move(step=3, player=1, x=7, y=6, timestamp=101.0),
            Move(step=4, player=2, x=0, y=0, timestamp=102.0),  # 模拟一步可能的“恶手”
            Move(step=5, player=1, x=7, y=5, timestamp=103.0)
        ]

        game_id = "test_visualization_001"

        # 2. 调用您的评分器
        result = self.scorer.score_moves(moves, game_id=game_id)

        # 3. === 验证返回值结构 (Keys) ===
        required_keys = ["score_curve", "critical_moments", "chart_path", "stats_summary", "csv_path"]
        for key in required_keys:
            self.assertIn(key, result, f"返回结果缺少关键字段: {key}")

        # 4. === 验证 Pandas 统计数据 ===
        stats = result["stats_summary"]
        print(f"📊 统计摘要: {stats}")
        self.assertIn("mean_score", stats)
        self.assertIn("brilliant_count", stats)
        # 确保算出来的是数字
        self.assertIsInstance(stats["mean_score"], float)

        # 5. === 验证文件是否真的生成了 ===
        # 验证 CSV (Pandas 导出)
        csv_path = result["csv_path"]
        self.assertTrue(os.path.exists(csv_path), f"CSV 文件未生成: {csv_path}")
        print(f"✅ CSV 文件已生成: {csv_path}")

        # 验证 PNG (Matplotlib 导出)
        chart_path = result["chart_path"]
        self.assertTrue(os.path.exists(chart_path), f"图表文件未生成: {chart_path}")
        print(f"✅ PNG 图表已生成: {chart_path}")

    def test_replay_save_load(self):
        """
        基础测试: 验证回放服务的保存读取
        """
        print("\n正在测试: 回放服务存取...")
        fake_game = GameReplay(
            game_id="test_io_001",
            start_time=datetime.now(),
            winner=1,
            moves=[Move(step=1, player=1, x=7, y=7, timestamp=100.0)]
        )

        # 保存
        path = self.replay_service.save_replay(fake_game)
        self.assertTrue(os.path.exists(path))

        # 读取
        loaded = self.replay_service.load_replay("test_io_001")
        self.assertEqual(loaded['game_id'], "test_io_001")
        print("✅ 存取功能正常")


if __name__ == '__main__':
    unittest.main()