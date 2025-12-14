import sys
import os
import random
import time
import math
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine.board import Board
from backend.engine.game_engine import GameEngine
from backend.ai.basic.classic_ai import GreedyAgent, RandomAgent
from stable_baselines3 import DQN

# --- FAST BENCHMARK OPPONENTS (Simplified for Speed) ---

def get_candidates(board, top_k):
    size = board.size
    moves = set()
    b = board.board
    occupied = False
    for x in range(size):
        for y in range(size):
            if b[x][y] != 0:
                occupied = True
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx==0 and dy==0: continue
                        nx, ny = x+dx, y+dy
                        if 0<=nx<size and 0<=ny<size and b[nx][ny] == 0:
                            moves.add((nx,ny))
    
    if not occupied:
        return [(7,7)]

    cand_list = list(moves)
    if not cand_list: return [(7,7)]
    
    # Sort by simple greedy value
    scored = []
    for mx, my in cand_list:
         val = quick_score_move(board, mx, my)
         scored.append((val, (mx, my)))
         
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

def quick_score_move(board, x, y):
    s = 0
    dirs = [(1,0), (0,1), (1,1), (1,-1)]
    for p in [1, 2]:
        board.board[x][y] = p
        max_len = 0
        for dx, dy in dirs:
            c = 1
            tx, ty = x+dx, y+dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == p: c+=1; tx+=dx; ty+=dy
            tx, ty = x-dx, y-dy
            while board.is_inside(tx, ty) and board.board[tx][ty] == p: c+=1; tx-=dx; ty-=dy
            max_len = max(max_len, c)
        board.board[x][y] = 0
        if max_len >= 5: s += 100000
        elif max_len == 4: s += 5000
        elif max_len == 3: s += 100
        elif max_len == 2: s += 10
    return s

def evaluate_state_fast(board, player):
    winner = board.get_game_result()
    if winner == player: return 100000
    if winner == (3-player): return -100000
    if winner == 3: return 0
    return random.random() * 5

class MinimaxAgent:
    def __init__(self, depth=1, width=3): 
        # Very shallow to be fast for "All Models" comparison
        self.depth = depth
        self.width = width
    
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return alpha_beta_search(board, player, self.depth, self.width)

def alpha_beta_search(board: Board, player: int, depth: int, width: int) -> Tuple[int, int]:
    candidates = get_candidates(board, width)
    best_val = -math.inf
    best_move = candidates[0] if candidates else (7,7)
    alpha = -math.inf
    beta = math.inf
    
    for mx, my in candidates:
        board.place_stone(mx, my, player)
        val = min_value(board, alpha, beta, depth - 1, player, width)
        board.board[mx][my] = 0
        if val > best_val:
            best_val = val
            best_move = (mx, my)
        alpha = max(alpha, best_val)
    return best_move

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

# --- MAIN EVALUATION ---

class FinalEvaluator:
    def __init__(self):
        self.models = {}
        self.opponents = {
            "Random": RandomAgent(),
            "Greedy": GreedyAgent(distance=2),
            "Minimax": MinimaxAgent(depth=1, width=3) # Weak Minimax
        }
        
    def load_models(self, path_list):
        for path in path_list:
            if os.path.exists(path):
                name = os.path.basename(path).replace(".zip", "")
                try:
                    self.models[name] = DQN.load(path)
                    print(f"Loaded {name}")
                except Exception as e:
                    print(f"Failed to load {name}: {e}")

    def run_tournament(self, games_per_match=10):
        print(f"\nTarget Games per Match: {games_per_match}")
        
        # DataFrame to store results
        records = []

        for model_name, model in self.models.items():
            print(f"\nEvaluating === {model_name} ===")
            for opp_name, opponent in self.opponents.items():
                print(f"  vs {opp_name}: ", end="", flush=True)
                wins = 0
                losses = 0
                draws = 0
                invalid = 0
                moves_log = []
                
                for i in range(games_per_match):
                    engine = GameEngine()
                    # Model is always Player 1 for consistency
                    model_p = 1
                    opp_p = 2
                    
                    while not engine.game_over:
                        if engine.current_player == model_p:
                            obs = np.array(engine.board.board, dtype=np.float32)
                            action, _ = model.predict(obs, deterministic=True)
                            x, y = divmod(int(action), 15)
                            
                            if not engine.board.is_valid_move(x, y):
                                engine.game_over = True
                                engine.winner = opp_p
                                invalid += 1
                                # print("X", end="", flush=True)
                            else:
                                engine.make_move(x, y)
                        else:
                            ox, oy = opponent.get_move(engine.board, opp_p)
                            if ox == -1:
                                engine.game_over = True
                                engine.winner = model_p
                            else:
                                engine.make_move(ox, oy)
                    
                    if engine.winner == model_p:
                        wins += 1
                        print("W", end="", flush=True)
                    elif engine.winner == opp_p:
                        losses += 1
                        print("L", end="", flush=True)
                    else:
                        draws += 1
                        print("D", end="", flush=True)
                    
                    moves_log.append(engine.board.move_count)
                
                win_rate = (wins / games_per_match) * 100
                avg_len = sum(moves_log) / len(moves_log) if moves_log else 0
                print(f" | WR: {win_rate:.1f}% | AvgMoves: {avg_len:.1f} | Invalid: {invalid}")
                
                records.append({
                    "Model": model_name,
                    "Opponent": opp_name,
                    "WinRate": win_rate,
                    "AvgMoves": avg_len,
                    "InvalidMoves": invalid
                })
        
        return pd.DataFrame(records)

if __name__ == "__main__":
    evaluator = FinalEvaluator()
    # List all candidate models
    candidates = [
        "models/dqn_gomoku_v2.zip",
        "models/dqn_gomoku_v3.zip",
        "models/dqn_gomoku_v4.zip",
        "models/dqn_15x15_final.zip"
    ]
    
    evaluator.load_models(candidates)
    
    if not evaluator.models:
        print("No models found to evaluate.")
    else:
        df = evaluator.run_tournament(games_per_match=10)
        print("\n\n====== FINAL COMPARISON REPORT ======")
        print(df.to_string(index=False))
        
        # Summary pivot
        print("\n--- Summary by Model ---")
        summary = df.groupby("Model")[["WinRate", "AvgMoves", "InvalidMoves"]].mean()
        print(summary)
        
        # Find best
        best_model = summary["WinRate"].idxmax()
        print(f"\nStrongest Model appears to be: {best_model}")
