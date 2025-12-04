from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Move(BaseModel):
    step: int = Field(..., description="第几手")
    player: int = Field(..., description="1=黑棋, 2=白棋")
    x: int = Field(..., description="横坐标")
    y: int = Field(..., description="纵坐标")
    timestamp: float = Field(..., description="时间戳")
    evaluation_score: Optional[float] = None # AI 评分
    is_critical: bool = False # 是否是关键手

class GameReplay(BaseModel):
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    winner: int
    moves: List[Move] = []
    stats: Optional[Dict] = None # 统计数据

    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game_001",
                "start_time": "2025-11-28T10:00:00",
                "winner": 1,
                "moves": []
            }
        }