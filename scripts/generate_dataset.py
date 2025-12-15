import sys
import os
import random
import time
import numpy as np
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine.board import Board
from backend.engine.game_engine import GameEngine
from backend.ai.basic.strong_ai import AlphaBetaAgent

from backend.ai.basic.classic_ai import GreedyAgent

def generate_dataset(num_games=100, save_path="data/sl_dataset.pkl"):
    """
    Generates a dataset of (BoardState, BestMove) pairs by pitting Strong AI against Greedy AI.
    Only saves moves from the Winner (to learn winning patterns).
    """
    print(f"Generating Anti-Greedy Dataset ({num_games} games)...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    teacher = AlphaBetaAgent(depth=2, time_limit=0.2) # Fast teacher
    dummy = GreedyAgent() # Punching bag
    
    dataset = [] 
    
    start_time = time.time()
    
    for i in range(num_games):
        engine = GameEngine()
        
        # Randomize start player
        # Even games: Teacher is P1
        # Odd games: Teacher is P2
        teacher_p = 1 if i % 2 == 0 else 2
        
        # Recording moves for this game: list of (player, input, outcome_move)
        game_history = []
        
        while not engine.game_over and engine.board.move_count < 225:
            p = engine.current_player
            
            # Prepare Input (Always from perspective of current player)
            board_input = np.zeros((15, 15), dtype=np.float32)
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
                # We don't learn from Greedy, unless we want to learn 'what not to do'? 
                # No, just learn winning moves.
            
            mx, my = move
            if mx == -1: break
            engine.make_move(mx, my)
            
        # Game Over
        winner = engine.winner
        # If teacher won, save teacher's moves
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
