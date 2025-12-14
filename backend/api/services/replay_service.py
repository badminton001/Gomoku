import json
import os
import glob
from typing import Dict, List, Optional, Any
from backend.analysis.replay import GameReplay
from backend.api.services.move_scorer import MoveScorer

class ReplayService:
    """
    游戏回放服务 (Person B)
    - 保存/加载 .json 录像
    - 调用 AI 对局势进行分析
    """

    def __init__(self, data_dir="data/games"):
        # 自动创建存储目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir
        # Lazy load scorer to avoid circular deps if any (though simple import is fine)
        self.scorer = None

    def save_replay(self, replay: GameReplay) -> str:
        """保存回放文件"""
        file_path = os.path.join(self.data_dir, f"{replay.game_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            # Check Pydantic version compatibility
            if hasattr(replay, 'model_dump_json'):
                f.write(replay.model_dump_json())
            else:
                f.write(replay.json())
        return file_path

    def load_replay(self, game_id: str) -> Optional[GameReplay]:
        """加载回放对象"""
        file_path = os.path.join(self.data_dir, f"{game_id}.json")
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return GameReplay(**data)

    def list_replays(self) -> List[Dict[str, Any]]:
        """列出所有保存的回放"""
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        replays = []
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract header info
                    replays.append({
                        "game_id": data.get("game_id"),
                        "start_time": data.get("start_time"),
                        "winner": data.get("winner"),
                        "moves_count": len(data.get("moves", []))
                    })
            except Exception:
                continue
        # Sort by time (newest first)
        replays.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return replays

    def analyze_replay(self, game_id: str) -> Dict[str, Any]:
        """
        对指定的回放进行 AI 分析
        """
        replay = self.load_replay(game_id)
        if not replay:
            return {"error": "Game not found"}
            
        if not self.scorer:
            self.scorer = MoveScorer(enable_mcts=False) # Default to generic scorer
            
        # Call Scorer
        # Note: Scorer expects List[Move] (from replay import Move)
        # GameReplay has .moves which is List[Move]
        analysis_result = self.scorer.score_moves(replay.moves, game_id=game_id)
        
        return analysis_result