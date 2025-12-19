import torch
import numpy as np
import time
import argparse
import sys
import os

sys.path.append(os.getcwd())
from backend.engine.game_engine import GameEngine
from backend.ai.baselines import RandomAgent, GreedyAgent
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.hybrid import HybridAgent

def print_board(engine):
    b = engine.board.board
    size = engine.board.size
    print("\n   " + " ".join([f"{i:X}" for i in range(size)]))
    for r in range(size):
        row_str = f"{r:X}  "
        for c in range(size):
            v = b[r][c]
            if v == 0: row_str += ". "
            elif v == 1: row_str += "X " # P1
            elif v == 2: row_str += "O " # P2
        print(row_str)
    print("")

def play_match(agent1, agent2, render=False):
    engine = GameEngine()
    engine.reset_game()
    start_t = time.time()
    
    while not engine.game_over:
        if render:
            print_board(engine)
            
        p = engine.current_player
        if p == 1:
            x, y = agent1.get_move(engine.board, p)
        else:
            x, y = agent2.get_move(engine.board, p)
            
        if not engine.make_move(x, y):
             print(f"Invalid move by {p} ({x},{y})")
             return 3 - p, engine.board.move_count, time.time() - start_t
             
    duration = time.time() - start_t
    if render:
        print_board(engine)
        print(f"Game Over in {duration:.1f}s. Winner: {engine.winner}")
        
    return engine.winner, engine.board.move_count, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs", choices=["greedy", "strong", "all"], default="strong")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    # Load Hybrid (P1)
    # Check for fine-tuned models in priority order
    possible_models = [
        "models/sl_policy_v2_kaggle.pth",     # Kaggle
        "models/sl_policy_v1_finetuned.pth",  # Finetuned
        "models/sl_policy_v1_base.pth"        # Base
    ]
    
    model_path = "models/sl_policy_v1_base.pth" # Default
    for p in possible_models:
        if os.path.exists(p):
            model_path = p
            print(f"Using Model: {model_path}")
            break
            
    hybrid_agent = HybridAgent(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    opponents = []
    if args.vs == "all":
        opponents = [("Greedy", GreedyAgent()), ("Strong", AlphaBetaAgent(depth=2, time_limit=1.0))]
    elif args.vs == "greedy":
        opponents = [("Greedy", GreedyAgent())]
    elif args.vs == "strong":
        opponents = [("Strong", AlphaBetaAgent(depth=2, time_limit=1.0))]
        
    print(f"\n>>> HYBRID AGENT TOURNAMENT (v1.0) <<<")
    print(f"Model: {hybrid_agent.model is not None}")
    
    for name, opp in opponents:
        print(f"\n--- vs {name} ---")
        wins = 0
        losses = 0
        draws = 0
        times = []
        
        for i in range(args.games):
            w, moves, t = play_match(hybrid_agent, opp, render=(args.games==1))
            if w == 1: wins += 1
            elif w == 2: losses += 1
            else: draws += 1
            times.append(t)
            print(f"Game {i+1}: {'WIN' if w==1 else 'LOSS' if w==2 else 'DRAW'} (Moves: {moves}, Time: {t:.1f}s)")
            
            # Save Logs
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"game_{i+1}_vs_{name}.txt")
            hybrid_agent.save_logs(log_path)
            # Clear logs
            hybrid_agent.logs = []
            
        avg_time = sum(times)/len(times) if times else 0
        print(f"Result vs {name}: {wins}-{losses}-{draws} (Avg Time: {avg_time:.1f}s)")
        
        if wins > losses:
            print(f"VERDICT: PASSED (Defeated {name})")
        else:
            print(f"VERDICT: FAILED (Lost to {name})")

if __name__ == "__main__":
    main()
