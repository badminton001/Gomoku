import json
import os
from typing import Dict
from backend.models.replay import GameReplay


class ReplayService:
    """
    游戏回放服务
    保存 .json 文件到 data/games/ 目录
    """

    def __init__(self, data_dir="data/games"):
        # 自动创建存储目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir

    def save_replay(self, replay: GameReplay) -> str:
        """
        保存回放文件
        """
        file_path = os.path.join(self.data_dir, f"{replay.game_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            # 使用 Pydantic 的 model_dump_json (v2) 或 json() (v1)
            f.write(replay.model_dump_json())
        return file_path

    def load_replay(self, game_id: str) -> Dict:
        """
        加载回放
        """
        file_path = os.path.join(self.data_dir, f"{game_id}.json")
        if not os.path.exists(file_path):
            return {"error": "Game not found"}

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)