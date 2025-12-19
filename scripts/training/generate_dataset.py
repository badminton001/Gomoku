import sys
import os
import random
import time
import numpy as np
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.engine.board import Board
from backend.engine.game_engine import GameEngine
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.baselines import GreedyAgent

def generate_dataset(num_games=100, save_path="data/sl_dataset.pkl"):
    """Generate Board/Move dataset."""
    print(f"Generating Anti-Greedy Dataset ({num_games} games)...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    teacher = AlphaBetaAgent(depth=2, time_limit=0.2) # Fast
    dummy = GreedyAgent()
    
    dataset = [] 
    
    start_time = time.time()
    
    for i in range(num_games):
        engine = GameEngine()
        
        # Randomize start player
        # Random start
        
        # Record records
        
        while not engine.game_over and engine.board.move_count < 225:
            p = engine.current_player
            
            # Prepare Input
            for r in range(15):
                for c in range(15):
                    v = engine.board.board[r][c]
                    if v == 0: board_input[r][c] = 0
                    elif v == p: board_input[r][c] = 1.0
                    else: board_input[r][c] = -1.0
            
            # Select Move
            if p == teacher_p:
                move = teacher.get_move(engine.board, p)
                game_history.append((p, board_input, move))
            else:
                move = dummy.get_move(engine.board, p)
                # Only teacher
            
            mx, my = move
            if mx == -1: break
            engine.make_move(mx, my)
            
        # Game Over
        winner = engine.winner
        # Save if teacher won
        if winner == teacher_p:
            for pl, inp, (mx, my) in game_history:
                if pl == teacher_p:
                    label = mx * 15 + my
                    dataset.append((inp, label))
            print(f"Game {i+1}: Teacher WON. Saved samples.")
        else:
            print(f"Game {i+1}: Teacher LOST/DRAW. Discarding.")
            
    print(f"Generation complete. {len(dataset)} samples collected in {time.time()-start_time:.1f}s.")
    
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    generate_dataset(num_games=20, save_path="data/anti_greedy.pkl")
