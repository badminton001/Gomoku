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
from backend.algorithms.classic_ai import GreedyAgent, RandomAgent
from stable_baselines3 import DQN

# --- Simplified AlphaBeta/Minimax for Benchmark Speed ---

def get_candidates(board, top_k):
    size = board.size
    moves = set()
    b = board.board
    # Fast scan for neighbors
    for x in range(size):
        for y in range(size):
            if b[x][y] != 0:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx==0 and dy==0: continue
                        nx, ny = x+dx, y+dy
                        if 0<=nx<size and 0<=ny<size and b[nx][ny] == 0:
                            moves.add((nx,ny))
    
    cand_list = list(moves)
    if not cand_list: return [(7,7)]
    
    # Sort by simple greedy value (atk + def) - SIMULATED GREEDY
    scored = []
    for mx, my in cand_list:
         val = quick_score_move(board, mx, my)
         scored.append((val, (mx, my)))
         
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

def quick_score_move(board, x, y):
    # Rough heuristic: sum of max consecutive stones in 4 dirs for both players
    # (Attack + Defense)
    s = 0
    dirs = [(1,0), (0,1), (1,1), (1,-1)]
    
    # Check if I play here (Assume Player 1)
    # Check if Opponent plays here (Block)
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
    return random.random() * 5 # Small noise to break ties

# --- Agents ---

class MinimaxAgent:
    def __init__(self, depth=2, width=4):
        self.depth = depth
        self.width = width
    
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        return alpha_beta_search(board, player, self.depth, self.width)

class AlphaBetaAgent:
    def __init__(self, depth=2, width=6):
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
        board.board[mx][my] = 0 # Undo
        
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


class Benchmarker:
    def __init__(self, model_path):
        try:
            print(f"Loading model from {model_path}...")
            self.model = DQN.load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

        self.agents = {
            "Random": RandomAgent(),
            "Greedy": GreedyAgent(distance=2),
            "Minimax-D2": MinimaxAgent(depth=2, width=4),
            "AlphaBeta-D2": AlphaBetaAgent(depth=2, width=6),
        }

    def run_benchmark(self, num_games=5):
        print(f"\nStarting Benchmark for V4 (5 games per opponent)...")
        results = {}

        for name, agent in self.agents.items():
            wins = 0
            losses = 0
            draws = 0
            moves_history = []
            
            print(f"\nVS {name}: ", end="", flush=True)
            
            for i in range(num_games):
                engine = GameEngine()
                v3_player = 1
                opp_player = 2
                
                while not engine.game_over:
                    if engine.current_player == v3_player:
                        obs = np.array(engine.board.board, dtype=np.float32)
                        action, _ = self.model.predict(obs, deterministic=True)
                        x, y = divmod(int(action), 15)
                        
                        if not engine.board.is_valid_move(x, y):
                             # Invalid move results in immediate loss
                             engine.game_over = True
                             engine.winner = opp_player
                             print("X", end="", flush=True) # X = Invalid
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
                    print("W", end="", flush=True)
                elif engine.winner == opp_player:
                    losses += 1
                    print("L", end="", flush=True)
                else:
                    draws += 1
                    print("D", end="", flush=True)
                
                moves_history.append(engine.board.move_count)
            
            avg_moves = sum(moves_history)/len(moves_history) if moves_history else 0
            rate = (wins / num_games) * 100
            results[name] = f"{rate}% ({wins}W-{losses}L-{draws}D, AvgMoves: {avg_moves:.1f})"
            print(f" -> {results[name]}")

        print("\n=== V4 Benchmark Summary ===")
        for k, v in results.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    path = "models/dqn_gomoku_v4.zip"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
    else:
        Benchmarker(path).run_benchmark(5)
