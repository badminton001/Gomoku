"""
Classic AI algorithms (Random, Greedy).
"""
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
    """Lightweight search statistics."""
    elapsed_ms: float
    explored_nodes: int
    candidate_moves: int


def load_ai_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def random_move(board: Board, distance: int = 2) -> Tuple[int, int]:
    """Return a random legal move near existing stones."""
    candidates = get_neighbor_moves(board, distance=distance)
    if not candidates:
        return (board.size // 2, board.size // 2)
        
    valid = [m for m in candidates if board.is_valid_move(m[0], m[1])]
    if not valid:
        # Fallback to full board scan
        empty = []
        for x in range(board.size):
            for y in range(board.size):
                if board.is_empty(x, y): empty.append((x,y))
        return random.choice(empty) if empty else (-1, -1)
        
    return random.choice(valid)


class RandomAgent:
    """Random move agent."""
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return random_move(board, distance=2)

class GreedyAgent:
    """1-Ply Greedy Agent (Local Evaluation)."""

    def __init__(self, distance: int = 2):
        self.distance = distance
        self.last_metrics: Optional[SearchMetrics] = None

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        start = time.perf_counter()
        candidates = get_neighbor_moves(board, distance=self.distance)
        if not candidates:
            return (board.size // 2, board.size // 2)

        best_score = -math.inf
        # Default to random among best
        best_move = candidates[0] 
        explored = 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        def evaluate_point(x: int, y: int, target: int) -> float:
            score = 0.0
            board.board[x][y] = target
            
            for dx, dy in directions:
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
                    
                # Basic Score
                if count >= 5: score += 100000
                elif count == 4: score += 5000
                elif count == 3: score += 1000
                elif count == 2: score += 100
                
            board.board[x][y] = 0
            return score

        # Evaluate Candidates
        opponent = 3 - player
        scored_moves = []
        
        for cx, cy in candidates:
            if not board.is_valid_move(cx, cy): 
                continue
                
            explored += 1
            attack = evaluate_point(cx, cy, player)
            defense = evaluate_point(cx, cy, opponent)
            
            # Simple heuristic: Attack + Defense bias
            total = attack + (defense * 0.9)
            scored_moves.append(((cx, cy), total))
            
            if total > best_score:
                best_score = total
                best_move = (cx, cy)
        
        # Randomize among top to avoid deterministic loops
        if scored_moves:
             top_moves = sorted(scored_moves, key=lambda x: x[1], reverse=True)
             if len(top_moves) > 3 and abs(top_moves[0][1] - top_moves[2][1]) < 10:
                  best_move = random.choice([m[0] for m in top_moves[:3]])
             else:
                  best_move = top_moves[0][0]

        elapsed = (time.perf_counter() - start) * 1000
        self.last_metrics = SearchMetrics(elapsed, explored, len(candidates))
        
        return best_move
