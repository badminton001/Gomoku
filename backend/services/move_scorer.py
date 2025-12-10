import os
import matplotlib.pyplot as plt
import pandas as pd  # <--- 1. 新增：引入 Pandas
from typing import List, Dict, Any
from backend.models.board import Board
from backend.models.replay import Move


class MoveScorer:

    def __init__(self):
        # 自动创建保存图片和数据的目录
        self.charts_dir = "data/charts"
        self.stats_dir = "data/stats"  # <--- 新增统计数据目录

        for d in [self.charts_dir, self.stats_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

    def score_moves(self, moves: List[Move], game_id: str = "temp") -> Dict[str, Any]:
        """
        分析整局游戏的着法质量，并生成 Pandas 数据表
        """
        critical_moments = []
        scores = []

        # 1. 模拟评分逻辑 (保持不变，后续对接 AI)
        import random
        for move in moves:
            # score = ai_model.evaluate(board, move)
            score = round(random.uniform(0, 1), 2)  # 占位符
            scores.append(score)

            # 识别关键时刻
            if score < 0.2:
                critical_moments.append({"step": move.step, "type": "Blunder (恶手)"})
            elif score > 0.8:
                critical_moments.append({"step": move.step, "type": "Brilliant (妙手)"})

        # --- 2. 新增：使用 Pandas 组织数据 ---
        # 把所有散乱的数据整合成一张表
        df = pd.DataFrame({
            "step": [m.step for m in moves],
            "player": [m.player for m in moves],
            "x": [m.x for m in moves],
            "y": [m.y for m in moves],
            "timestamp": [m.timestamp for m in moves],
            "ai_score": scores,
            "move_type": ["Normal"] * len(moves)  # 默认为普通
        })

        # 标记关键手到 DataFrame 中
        df.loc[df['ai_score'] > 0.8, 'move_type'] = 'Brilliant'
        df.loc[df['ai_score'] < 0.2, 'move_type'] = 'Blunder'

        # 计算统计指标
        stats_summary = {
            "mean_score": round(df['ai_score'].mean(), 3),
            "std_dev": round(df['ai_score'].std(), 3),
            "brilliant_count": int((df['ai_score'] > 0.8).sum()),
            "blunder_count": int((df['ai_score'] < 0.2).sum())
        }

        # 保存为 CSV (可选，方便用 Excel 打开查看)
        csv_path = os.path.join(self.stats_dir, f"{game_id}_stats.csv")
        df.to_csv(csv_path, index=False)

        # --- 3. 画图 (调用之前的逻辑) ---
        chart_path = self.generate_analysis_chart(scores, game_id)

        return {
            "score_curve": scores,
            "critical_moments": critical_moments,
            "chart_path": chart_path,
            "stats_summary": stats_summary,  # 返回统计摘要
            "csv_path": csv_path  # 返回 CSV 文件路径
        }

    def generate_analysis_chart(self, scores: List[float], game_id: str) -> str:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='#1f77b4', label='Win Rate')
        plt.title(f'Game Analysis: {game_id}', fontsize=14)
        plt.xlabel('Move Step', fontsize=12)
        plt.ylabel('AI Evaluation Score', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        for i, score in enumerate(scores):
            step = i + 1
            if score > 0.8:
                plt.annotate('!', (step, score), textcoords="offset points", xytext=(0, 10), ha='center', color='green',
                             fontweight='bold')
            elif score < 0.2:
                plt.annotate('?', (step, score), textcoords="offset points", xytext=(0, -10), ha='center', color='red',
                             fontweight='bold')
        output_filename = f"{game_id}_analysis.png"
        output_path = os.path.join(self.charts_dir, output_filename)
        plt.savefig(output_path)
        plt.close()
        return output_path