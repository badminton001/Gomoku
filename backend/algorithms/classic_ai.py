import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.algorithms.mcts_ai import get_neighbor_moves
from backend.models.board import Board


@dataclass
class SearchMetrics:
    """Lightweight search statistics for benchmarking."""

    elapsed_ms: float
    explored_nodes: int
    candidate_moves: int


def load_ai_config(path: str) -> Dict[str, Any]:
    """
    Load hyperparameter configuration from a JSON file.
    YAML can be supported by converting to JSON or installing PyYAML, but JSON keeps dependencies minimal.
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
    return random.choice(candidates)


def _sequence_score(length: int, open_ends: int) -> int:
    """Score a contiguous sequence with given open ends."""
    if length >= 5:
        return 1_000_000
    if length == 4:
        return 100_000 if open_ends == 2 else 10_000
    if length == 3:
        return 5_000 if open_ends == 2 else 500
    if length == 2:
        return 200 if open_ends == 2 else 50
    if length == 1:
        return 10 if open_ends == 2 else 2
    return 0


def evaluate_board(board: Board, player: int) -> float:
    """
    Heuristic evaluation based on run length and openness.
    Positive values favor `player`, negative values favor the opponent.
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def score_for(target: int) -> int:
        score = 0
        for x in range(board.size):
            for y in range(board.size):
                if board.board[x][y] != target:
                    continue
                for dx, dy in directions:
                    prev_x, prev_y = x - dx, y - dy
                    if board.is_inside(prev_x, prev_y) and board.board[prev_x][prev_y] == target:
                        # Skip if this is not the start of the sequence
                        continue

                    length = 0
                    cx, cy = x, y
                    while board.is_inside(cx, cy) and board.board[cx][cy] == target:
                        length += 1
                        cx += dx
                        cy += dy

                    open_ends = 0
                    if board.is_inside(prev_x, prev_y) and board.is_empty(prev_x, prev_y):
                        open_ends += 1
                    if board.is_inside(cx, cy) and board.is_empty(cx, cy):
                        open_ends += 1

                    score += _sequence_score(length, open_ends)
        return score

    player_score = score_for(player)
    opponent_score = score_for(3 - player)
    # Slightly penalize opponent to emphasize defense
    return player_score - 1.1 * opponent_score


class GreedyAgent:
    """
    One-ply greedy agent that picks the move with the best heuristic score.
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
        best_move = candidates[0]
        explored = 0

        for move in candidates:
            if not board.is_valid_move(move[0], move[1]):
                continue
            explored += 1
            board.place_stone(move[0], move[1], player)
            score = evaluate_board(board, player)
            board.board[move[0]][move[1]] = 0
            board.move_count -= 1

            if score > best_score:
                best_score = score
                best_move = move

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.last_metrics = SearchMetrics(
            elapsed_ms=elapsed_ms, explored_nodes=explored, candidate_moves=len(candidates)
        )
        return best_move


class MinimaxAgent:
    """Depth-limited minimax search without pruning."""

    def __init__(self, depth: int = 2, distance: int = 2, candidate_limit: Optional[int] = 12):
        self.depth = depth
        self.distance = distance
        self.candidate_limit = candidate_limit
        self.last_metrics: Optional[SearchMetrics] = None
        self._nodes = 0

    def _ordered_candidates(self, board: Board, player: int) -> List[Tuple[int, int]]:
        moves = get_neighbor_moves(board, distance=self.distance)
        scored_moves = []
        opponent = 3 - player
        for mv in moves:
            if not board.is_valid_move(mv[0], mv[1]):
                continue

            block_bonus = 0
            if board.place_stone(mv[0], mv[1], opponent):
                if board.get_game_result() == opponent:
                    block_bonus = 900_000
                board.board[mv[0]][mv[1]] = 0
                board.move_count -= 1

            board.place_stone(mv[0], mv[1], player)
            score = evaluate_board(board, player) + block_bonus
            board.board[mv[0]][mv[1]] = 0
            board.move_count -= 1
            scored_moves.append((score, mv))
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        ordered = [mv for _, mv in scored_moves]
        if self.candidate_limit is not None:
            ordered = ordered[: self.candidate_limit]
        if not ordered:
            ordered.append((board.size // 2, board.size // 2))
        return ordered

    def _minimax(
        self, board: Board, depth: int, current_player: int, max_player: int
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        self._nodes += 1
        result = board.get_game_result()
        if result == max_player:
            return 1_000_000.0, None
        if result == 3 - max_player:
            return -1_000_000.0, None
        if result == 3:
            return 0.0, None
        if depth == 0:
            return evaluate_board(board, max_player), None

        candidates = self._ordered_candidates(board, current_player)

        best_move: Optional[Tuple[int, int]] = None
        if current_player == max_player:
            best_value = -math.inf
            for move in candidates:
                if not board.place_stone(move[0], move[1], current_player):
                    continue
                value, _ = self._minimax(board, depth - 1, 3 - current_player, max_player)
                board.board[move[0]][move[1]] = 0
                board.move_count -= 1
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move

        # Minimizing opponent
        best_value = math.inf
        for move in candidates:
            if not board.place_stone(move[0], move[1], current_player):
                continue
            value, _ = self._minimax(board, depth - 1, 3 - current_player, max_player)
            board.board[move[0]][move[1]] = 0
            board.move_count -= 1
            if value < best_value:
                best_value = value
                best_move = move
        return best_value, best_move

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        self._nodes = 0
        start = time.perf_counter()
        _, move = self._minimax(board, self.depth, player, player)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if move is None:
            move = random_move(board, distance=self.distance)
        self.last_metrics = SearchMetrics(
            elapsed_ms=elapsed_ms,
            explored_nodes=self._nodes,
            candidate_moves=len(get_neighbor_moves(board, self.distance)),
        )
        return move


class AlphaBetaAgent:
    """Minimax with alpha-beta pruning and lightweight caching."""

    def __init__(
        self,
        depth: int = 3,
        distance: int = 2,
        candidate_limit: Optional[int] = 12,
        use_eval_cache: bool = True,
    ):
        self.depth = depth
        self.distance = distance
        self.candidate_limit = candidate_limit
        self.use_eval_cache = use_eval_cache
        self.last_metrics: Optional[SearchMetrics] = None
        self._nodes = 0
        self._eval_cache: Dict[Tuple[str, int], float] = {}

    def _cached_eval(self, board: Board, player: int) -> float:
        if not self.use_eval_cache:
            return evaluate_board(board, player)
        key = (board.to_string(), player)
        if key not in self._eval_cache:
            self._eval_cache[key] = evaluate_board(board, player)
        return self._eval_cache[key]

    def _ordered_candidates(self, board: Board, player: int, depth: int) -> List[Tuple[int, int]]:
        moves = get_neighbor_moves(board, distance=self.distance)
        scored = []
        opponent = 3 - player
        for mv in moves:
            if not board.is_valid_move(mv[0], mv[1]):
                continue

            block_bonus = 0
            if board.place_stone(mv[0], mv[1], opponent):
                if board.get_game_result() == opponent:
                    block_bonus = 900_000
                board.board[mv[0]][mv[1]] = 0
                board.move_count -= 1

            board.place_stone(mv[0], mv[1], player)
            score = self._cached_eval(board, player) + block_bonus
            board.board[mv[0]][mv[1]] = 0
            board.move_count -= 1
            scored.append((score, mv))
        scored.sort(key=lambda x: x[0], reverse=True)

        ordered = [mv for _, mv in scored]
        limit = self.candidate_limit
        if limit is not None and depth > 1:
            # Slightly tighten the branching factor below root for speed.
            limit = max(6, int(limit * 0.8))
        if limit is not None:
            ordered = ordered[:limit]
        if not ordered:
            ordered.append((board.size // 2, board.size // 2))
        return ordered

    def _evaluate_terminal(self, board: Board, max_player: int) -> float:
        result = board.get_game_result()
        if result == max_player:
            return 1_000_000.0
        if result == 3 - max_player:
            return -1_000_000.0
        if result == 3:
            return 0.0
        return None  # type: ignore[return-value]

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        current_player: int,
        max_player: int,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        self._nodes += 1
        terminal_value = self._evaluate_terminal(board, max_player)
        if terminal_value is not None:
            return terminal_value, None
        if depth == 0:
            return self._cached_eval(board, max_player), None

        candidates = self._ordered_candidates(board, current_player, depth)
        best_move: Optional[Tuple[int, int]] = None

        if current_player == max_player:
            value = -math.inf
            for move in candidates:
                if not board.place_stone(move[0], move[1], current_player):
                    continue
                result = self._evaluate_terminal(board, max_player)
                if result is None:
                    child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, 3 - current_player, max_player)
                else:
                    child_value = result
                board.board[move[0]][move[1]] = 0
                board.move_count -= 1
                if child_value > value:
                    value = child_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move

        value = math.inf
        for move in candidates:
            if not board.place_stone(move[0], move[1], current_player):
                continue
            result = self._evaluate_terminal(board, max_player)
            if result is None:
                child_value, _ = self._alphabeta(board, depth - 1, alpha, beta, 3 - current_player, max_player)
            else:
                child_value = result
            board.board[move[0]][move[1]] = 0
            board.move_count -= 1
            if child_value < value:
                value = child_value
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        self._nodes = 0
        self._eval_cache = {}
        start = time.perf_counter()
        _, move = self._alphabeta(board, self.depth, -math.inf, math.inf, player, player)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if move is None:
            move = random_move(board, distance=self.distance)
        self.last_metrics = SearchMetrics(
            elapsed_ms=elapsed_ms,
            explored_nodes=self._nodes,
            candidate_moves=len(get_neighbor_moves(board, self.distance)),
        )
        return move
