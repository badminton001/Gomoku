"""
Replay Data Models.

Defines Pydantic models for game moves and replay metadata.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Move(BaseModel):
    step: int = Field(..., description="Move number")
    player: int = Field(..., description="1=Black, 2=White")
    x: int = Field(..., description="Coordinate X")
    y: int = Field(..., description="Coordinate Y")
    timestamp: float = Field(..., description="Timestamp")
    evaluation_score: Optional[float] = None
    is_critical: bool = False

class GameReplay(BaseModel):
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    winner: int
    moves: List[Move] = []
    stats: Optional[Dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "game_id": "game_001",
                "start_time": "2025-11-28T10:00:00",
                "winner": 1,
                "moves": []
            }
        }