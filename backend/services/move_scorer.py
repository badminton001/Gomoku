import os
import matplotlib.pyplot as plt  # 引入画图库
from typing import List, Dict, Any
from backend.models.board import Board
from backend.models.replay import Move


class MoveScorer:

    def __init__(self):
        # 自动创建保存图片的目录
        self.charts_dir = "data/charts"
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)

    def score_moves(self, moves: List[Move], game_id: str = "temp") -> Dict[str, Any]:
        """
        分析整局游戏的着法质量
        """
        critical_moments = []
        scores = []

        # 模拟评分逻辑 (对接 Person C)
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

        # <--- 2. 新增：调用下面的画图函数，生成图片 --->
        chart_path = self.generate_analysis_chart(scores, game_id)

        return {
            "score_curve": scores,
            "critical_moments": critical_moments,
            "chart_path": chart_path  # <--- 返回图片的路径给前端
        }

   #画图函数
    def generate_analysis_chart(self, scores: List[float], game_id: str) -> str:
        """
        生成胜率波动图，并保存为图片文件
        返回：图片文件的路径
        """
        # 设置画布大小
        plt.figure(figsize=(10, 5))

        # 画折线图 (x轴是步数，y轴是分数)
        plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='#1f77b4', label='Win Rate')

        # 设置标题和标签
        plt.title(f'Game Analysis: {game_id}', fontsize=14)
        plt.xlabel('Move Step', fontsize=12)
        plt.ylabel('AI Evaluation Score', fontsize=12)
        plt.ylim(0, 1)  # 分数范围固定在 0-1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 标记关键手 (大于0.8或小于0.2的点)
        for i, score in enumerate(scores):
            step = i + 1
            if score > 0.8:
                plt.annotate('!', (step, score), textcoords="offset points", xytext=(0, 10), ha='center', color='green',
                             fontweight='bold')
            elif score < 0.2:
                plt.annotate('?', (step, score), textcoords="offset points", xytext=(0, -10), ha='center', color='red',
                             fontweight='bold')

        # 保存图片到 data/charts 目录
        output_filename = f"{game_id}_analysis.png"
        output_path = os.path.join(self.charts_dir, output_filename)

        plt.savefig(output_path)
        plt.close()  # 以此释放内存

        print(f"Chart generated at: {output_path}")
        return output_path