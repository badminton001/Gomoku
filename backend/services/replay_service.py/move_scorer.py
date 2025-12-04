from typing import List, Dict, Any
from backend.models.board import Board
from backend.models.replay import Move


class MoveScorer:

    def __init__(self):
        pass

    def score_moves(self, moves: List[Move]) -> Dict[str, Any]:
        """
        分析整局游戏的着法质量
        """
        critical_moments = []
        scores = []

        # 模拟评分逻辑
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

        return {
            "score_curve": scores,  # 供前端画图
            "critical_moments": critical_moments
        }