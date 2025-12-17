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

# Corrected Imports from refactored codebase
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.baselines import GreedyAgent, SearchMetrics, random_move
from backend.ai.mcts import get_neighbor_moves
from backend.engine.board import Board

def load_ai_config(path: str):
    """Safely load config or return defaults."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: Config not found, using defaults")
        return {
            "greedy": {"distance": 2},
            "minimax": {"depth": 2, "distance": 2, "candidate_limit": 12},
            "alpha_beta": {"depth": 3, "distance": 2, "candidate_limit": 10, "use_eval_cache": True}
        }

# Helper wrapper for Random to match interface
class RandomWrapper:
    def __init__(self):
        self.last_metrics = None
    def get_move(self, board, player):
        return random_move(board)

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
    elif hasattr(agent, 'last_metrics') and agent.last_metrics:
        metrics = agent.last_metrics
    else:
        # Fallback if agent doesn't log metrics
        metrics = SearchMetrics(elapsed, 0, 0)

    # Estimate quality using AlphaBeta's evaluator for consistency
    evaluator = AlphaBetaAgent(depth=1)
    if b.is_valid_move(move[0], move[1]):
        b.place_stone(move[0], move[1], player)
        quality = evaluator.evaluate_board(b, player)
    else:
        quality = float("-inf")

    return {
        "agent": agent_name,
        "move": move,
        "time_ms": round(elapsed, 3), # Use measured elapsed for all
        "candidate_moves": metrics.candidate_moves,
        "explored_nodes": metrics.explored_nodes,
        "quality": quality,
    }


def run_benchmark() -> List[Dict]:
    random.seed(42)
    cfg = load_ai_config("config/ai_config.json")

    base_board = Board(size=15) # Standard size
    # Mid-game snapshot (Center area)
    positions = [
        (7, 7, 1),
        (7, 8, 2),
        (8, 8, 1),
        (9, 8, 2),
        (8, 7, 1),
        (6, 8, 2),
        (7, 6, 1),
        (6, 7, 2),
    ]
    for x, y, p in positions:
        base_board.place_stone(x, y, p)

    greedy = GreedyAgent(distance=cfg["greedy"]["distance"])
    
    # Simulate Minimax using AlphaBeta with depth 2
    minimax = AlphaBetaAgent(
        depth=cfg["minimax"]["depth"],
        time_limit=5.0
    )
    
    alphabeta = AlphaBetaAgent(
        depth=cfg["alpha_beta"]["depth"],
        time_limit=5.0
    )

    random_agent = RandomWrapper()

    print("Running benchmark...")
    results = [
        _measure("Random", base_board, 1, agent=random_agent),
        _measure("Greedy", base_board, 1, greedy),
        _measure("Minimax", base_board, 1, minimax),
        _measure("AlphaBeta", base_board, 1, alphabeta),
    ]
    return results


if __name__ == "__main__":
    res = run_benchmark()
    # Print as Markdown Table
    print(f"\n| {'Agent':<12} | {'Time (ms)':<10} | {'Nodes':<8} | {'Score':<10} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*10}|{'-'*12}|")
    for r in res:
        print(f"| {r['agent']:<12} | {r['time_ms']:<10} | {r['explored_nodes']:<8} | {r['quality']:<10} |")

