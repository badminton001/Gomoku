import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.ai.advanced.mcts_ai import get_neighbor_moves
from backend.engine.board import Board


@dataclass
class SearchMetrics:
    """Lightweight search statistics for benchmarking."""

    elapsed_ms: float
    explored_nodes: int
    candidate_moves: int


def load_ai_config(path: str) -> Dict[str, Any]:
    """
    Load hyperparameter configuration from a JSON file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def random_move(board: Board, distance: int = 2) -> Tuple[int, int]:
    """
    Return a random legal move near existing stones (distance-limited neighborhood).
    Falls back to board center if nothing is available.
    """
    candidates = get_neighbor_moves(board, distance=distance)
    if not candidates:
        center = (board.size // 2, board.size // 2)
        return center
    # Filter valid
    valid = [m for m in candidates if board.is_valid_move(m[0], m[1])]
    if not valid:
        # Try full board random
        empty = []
        for x in range(board.size):
            for y in range(board.size):
                if board.is_empty(x, y): empty.append((x,y))
        if empty: return random.choice(empty)
        return (7,7)
        
    return random.choice(valid)


class RandomAgent:
    """Simply picks a random valid move nearby."""
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return random_move(board, distance=2)

class GreedyAgent:
    """
    Optimized One-ply greedy agent using local evaluation.
    """

    def __init__(self, distance: int = 2):
        self.distance = distance
        self.last_metrics: Optional[SearchMetrics] = None

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        start = time.perf_counter()
        candidates = get_neighbor_moves(board, distance=self.distance)
        if not candidates:
            return (board.size // 2, board.size // 2)

        best_score = -math.inf
        # Default to random choice among candidates if all scores equal
        best_move = candidates[0] 
        explored = 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def evaluate_point(x: int, y: int, target: int) -> float:
            # Score of chains formed by 'target' at (x,y)
            score = 0.0
            # Temporarily place stone to check offense
            board.board[x][y] = target
            
            for dx, dy in directions:
                # Count consecutive stones
                count = 1
                # Forward
                tx, ty = x + dx, y + dy
                while board.is_inside(tx, ty) and board.board[tx][ty] == target:
                    count += 1
                    tx += dx
                    ty += dy
                # Backward
                tx, ty = x - dx, y - dy
                while board.is_inside(tx, ty) and board.board[tx][ty] == target:
                    count += 1
                    tx -= dx
                    ty -= dy
                
                # Check openness (simple check)
                # (Omitted logic for brevity in this sync, using simplified scoring)
                
                if count >= 5:
                    score += 100000.0
                elif count == 4:
                    score += 10000.0
                elif count == 3:
                    score += 1000.0 
                elif count == 2:
                     score += 100.0
            
            board.board[x][y] = 0 # Restore
            return score

        # Filter valid candidates first
        valid_candidates = []
        for mx, my in candidates:
            if board.is_valid_move(mx, my):
                valid_candidates.append((mx, my))
        
        if not valid_candidates:
             return (board.size // 2, board.size // 2)

        for mx, my in valid_candidates:
            explored += 1
            
            # 1. Offense Score (My potential gain)
            attack_score = evaluate_point(mx, my, player)
            
            # 2. Defense Score (Opponent's potential gain if I don't block)
            defense_score = evaluate_point(mx, my, 3 - player)
            
            final_score = attack_score + defense_score 
            
            if final_score > best_score:
                best_score = final_score
                best_move = (mx, my)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_metrics = SearchMetrics(
            elapsed_ms=elapsed_ms, explored_nodes=explored, candidate_moves=len(candidates)
        )
        return best_move
