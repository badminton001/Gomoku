import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.algorithms.classic_ai import (  # noqa: E402
    AlphaBetaAgent,
    GreedyAgent,
    MinimaxAgent,
    SearchMetrics,
    evaluate_board,
    load_ai_config,
    random_move,
)
from backend.algorithms.mcts_ai import get_neighbor_moves  # noqa: E402
from backend.models.board import Board  # noqa: E402


def _measure(agent_name: str, board: Board, player: int, agent) -> Dict:
    b = copy.deepcopy(board)
    start = time.perf_counter()
    move = agent.get_move(b, player)
    elapsed = (time.perf_counter() - start) * 1000

    if agent_name == "Random":
        # Synthesize metrics for the random baseline
        metrics = SearchMetrics(
            elapsed_ms=elapsed,
            explored_nodes=1,
            candidate_moves=len(get_neighbor_moves(board)),
        )
    else:
        metrics = agent.last_metrics

    # Estimate quality as heuristic after the chosen move
    if b.is_valid_move(move[0], move[1]):
        b.place_stone(move[0], move[1], player)
        quality = evaluate_board(b, player)
    else:
        quality = float("-inf")

    return {
        "agent": agent_name,
        "move": move,
        "time_ms": round(metrics.elapsed_ms, 3),
        "candidate_moves": metrics.candidate_moves,
        "explored_nodes": metrics.explored_nodes,
        "quality": quality,
    }


def run_benchmark() -> List[Dict]:
    random.seed(42)
    cfg = load_ai_config("backend/config/ai_config.json")

    base_board = Board(size=10)
    # Mid-game snapshot
    positions = [
        (5, 5, 1),
        (5, 6, 2),
        (6, 6, 1),
        (7, 6, 2),
        (6, 5, 1),
        (4, 6, 2),
        (5, 4, 1),
        (4, 5, 2),
    ]
    for x, y, p in positions:
        base_board.place_stone(x, y, p)

    greedy = GreedyAgent(distance=cfg["greedy"]["distance"])
    minimax = MinimaxAgent(
        depth=cfg["minimax"]["depth"],
        distance=cfg["minimax"]["distance"],
        candidate_limit=cfg["minimax"]["candidate_limit"],
    )
    alphabeta = AlphaBetaAgent(
        depth=cfg["alpha_beta"]["depth"],
        distance=cfg["alpha_beta"]["distance"],
        candidate_limit=cfg["alpha_beta"]["candidate_limit"],
        use_eval_cache=cfg["alpha_beta"].get("use_eval_cache", True),
    )

    random_wrapper = type("RandomWrap", (), {"get_move": lambda self, b, p: random_move(b)})()

    results = [
        _measure("Random", base_board, 1, agent=random_wrapper),
        _measure("Greedy", base_board, 1, greedy),
        _measure("Minimax", base_board, 1, minimax),
        _measure("AlphaBeta", base_board, 1, alphabeta),
    ]
    return results


if __name__ == "__main__":
    res = run_benchmark()
    print(json.dumps(res, indent=2))
