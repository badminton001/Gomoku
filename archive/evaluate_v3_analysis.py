import sys
import os
import random
import time
import math
import copy
import numpy as np
from typing import List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.board import Board
from backend.models.game_engine import GameEngine
from backend.algorithms.classic_ai import GreedyAgent
from stable_baselines3 import DQN

# --- Simplified AlphaBeta/Minimax ---
# Since Python is slow, we use a very shallow search with aggressive pruning/ordering
# relying on the GreedyAgent's heuristic for leaf evaluation.

class MinimaxAgent:
    """
    Simulated Minimax/AlphaBeta Agent.
    Actually uses Alpha-Beta Pruning because pure Minimax is too slow.
    Differences from 'AlphaBetaAgent': 
    - Smaller search width/depth to represent a 'weaker' lookahead.
    """
    def __init__(self, depth=2, width=4):
        self.depth = depth
        self.width = width
    
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return alpha_beta_search(board, player, self.depth, self.width)

class AlphaBetaAgent:
    """
    Stronger Alpha-Beta Agent.
    """
    def __init__(self, depth=2, width=8):
        self.depth = depth
        self.width = width

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return alpha_beta_search(board, player, self.depth, self.width)

# Helper function
def alpha_beta_search(board: Board, player: int, depth: int, width: int) -> Tuple[int, int]:
    # 1. Get Candidates (Heuristic Ordering)
    # We use a custom simple neighbor generator for speed
    candidates = get_candidates(board, width)
    
    best_val = -math.inf
    best_move = candidates[0] if candidates else (7,7)
    
    alpha = -math.inf
    beta = math.inf
    
    for mx, my in candidates:
        board.place_stone(mx, my, player)
        val = min_value(board, alpha, beta, depth - 1, player, width)
        board.board[mx][my] = 0 # Undo
        
        if val > best_val:
            best_val = val
            best_move = (mx, my)
        alpha = max(alpha, best_val)
        
    return best_move

def max_value(board, alpha, beta, depth, player, width):
    if depth == 0 or board.get_game_result() != 0:
        return evaluate_state_fast(board, player)
        
    candidates = get_candidates(board, width)
    v = -math.inf
    for mx, my in candidates:
        board.place_stone(mx, my, player)
        v = max(v, min_value(board, alpha, beta, depth - 1, player, width))
        board.board[mx][my] = 0
        if v >= beta: return v
        alpha = max(alpha, v)
    return v

def min_value(board, alpha, beta, depth, player, width):
    opponent = 3 - player
    if depth == 0 or board.get_game_result() != 0:
        return evaluate_state_fast(board, player)
        
    candidates = get_candidates(board, width)
    v = math.inf
    for mx, my in candidates:
        board.place_stone(mx, my, opponent)
        v = min(v, max_value(board, alpha, beta, depth - 1, player, width))
        board.board[mx][my] = 0
        if v <= alpha: return v
        beta = min(beta, v)
    return v

def get_candidates(board, top_k):
    # Fast neighbor finding
    size = board.size
    moves = set()
    b = board.board
    # Simply scan board for non-empty
    # Optimization: maintain a list of occupied cells in game_engine? 
    # For now, just scan 15x15 is okay-ish (225 iters)
    for x in range(size):
        for y in range(size):
            if b[x][y] != 0:
                # Add neighbors
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx==0 and dy==0: continue
                        nx, ny = x+dx, y+dy
                        if 0<=nx<size and 0<=ny<size and b[nx][ny] == 0:
                            moves.add((nx,ny))
    
    cand_list = list(moves)
    if not cand_list: return [(7,7)]
    
    # Sort by simple heuristic (Greedy-lite)
    # Check 1-step immediate value
    scored = []
    
    # Determine 'current' player for scoring - doesn't matter much for ordering, 
    # just want relevant moves. Let's assume we are ordering for the 'next' player.
    # But get_candidates is used by Min and Max.
    # It's better to sort by "Activity" (creates lines or blocks lines).
    
    for mx, my in cand_list:
         s = 0
         # Quick score: sum of adjacent stones? 
         # Or use the quick_score from before
         # Score for Player 1 (Black)
         s += quick_score_point(board, mx, my, 1)
         # Score for Player 2 (White) -> Blocking value
         s += quick_score_point(board, mx, my, 2)
         scored.append((s, (mx, my)))
         
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

def quick_score_point(board, x, y, player):
    # Check 4 directions for consecutive stones
    score = 0
    dirs = [(1,0), (0,1), (1,1), (1,-1)]
    for dx, dy in dirs:
        c = 0
        # Check adjacent
        if board.is_inside(x+dx, y+dy) and board.board[x+dx][y+dy] == player: c+=1
        if board.is_inside(x-dx, y-dy) and board.board[x-dx][y-dy] == player: c+=1
        if c > 0: score += (10**c)
    return score

def evaluate_state_fast(board, player):
    winner = board.get_game_result()
    if winner == player: return 100000
    if winner == (3-player): return -100000
    if winner == 3: return 0
    
    # Heuristic: Count active 3s and 4s
    # Very simplified: Random noise + piece count difference to break ties
    # (Real eval requires complex logic, omitted for speed)
    return random.random() * 10 

class Benchmarker:
    def __init__(self, v3_model_path):
        self.model = DQN.load(v3_model_path)
        self.agents = {
            "Greedy": GreedyAgent(distance=2),
            "Minimax-D2": MinimaxAgent(depth=2, width=4),
            "AlphaBeta-D2": AlphaBetaAgent(depth=2, width=6),
        }

    def run_benchmark(self, num_games=5): # 5 games each
        print(f"Starting Benchmark ({num_games} games per opponent)...")
        
        for name, agent in self.agents.items():
            wins = 0
            for i in range(num_games):
                engine = GameEngine()
                v3_player = 1
                opp_player = 2
                
                print(f"[{name} Game {i+1}]", end=" ", flush=True)
                
                while not engine.game_over:
                    if engine.current_player == v3_player:
                        obs = np.array(engine.board.board, dtype=np.float32)
                        action, _ = self.model.predict(obs, deterministic=True)
                        x, y = divmod(int(action), 15)
                        if not engine.board.is_valid_move(x, y):
                             engine.game_over = True; engine.winner = opp_player
                             print("X", end="", flush=True) # Invalid
                        else:
                             engine.make_move(x, y)
                    else:
                        ox, oy = agent.get_move(engine.board, opp_player)
                        if ox == -1: 
                             engine.game_over = True; engine.winner = v3_player
                        else:
                             engine.make_move(ox, oy)
                
                if engine.winner == v3_player: 
                    wins += 1
                    print("WIN", end=" ")
                elif engine.winner == opp_player:
                    print("LOSE", end=" ")
                else:
                    print("DRAW", end=" ")
                print(f"({engine.board.move_count} moves)")
            
            print(f">>> V3 vs {name}: {wins}/{num_games} ({wins/num_games*100:.1f}%)")

if __name__ == "__main__":
    if os.path.exists("models/dqn_gomoku_v3.zip"):
        Benchmarker("models/dqn_gomoku_v3.zip").run_benchmark(5)
    elif os.path.exists("models/dqn_gomoku_v2.zip"):
        print("V3 not found, testing V2...")
        Benchmarker("models/dqn_gomoku_v2.zip").run_benchmark(5)
    else:
        print("No models found!")
