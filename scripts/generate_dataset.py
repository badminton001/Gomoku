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

def generate_dataset(num_games=50, save_path="data/sl_dataset.pkl"):
    """
    Generates a dataset of (BoardState, BestMove) pairs using Strong AI.
    """
    print(f"Generating dataset with {num_games} games...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Teacher: Strong AlphaBeta Agent
    # Depth 2 is fast and decent (~infinite ELO vs Random). 
    # Depth 4 is stronger but slower. For "Fast" data gen, we use Depth 2 mixed with Depth 3.
    teacher = AlphaBetaAgent(depth=2, time_limit=1.0)
    
    dataset = [] # List of (board_matrix, move_index)
    
    start_time = time.time()
    total_moves = 0
    
    for i in range(num_games):
        engine = GameEngine()
        game_moves = 0
        
        # Add some randomness to openings so games are different
        # First 2 moves are random near center
        engine.make_move(7, 7) # Center
        
        # Random 2nd move
        candidates = [(6,6), (6,7), (6,8), (7,6), (7,8), (8,6), (8,7), (8,8)]
        rx, ry = random.choice(candidates)
        engine.make_move(rx, ry)
        
        while not engine.game_over and engine.board.move_count < 225:
            # Current board state for input
            # We need to canonicalize perspective? 
            # For simplicity: Input = Board where 1=MyStone, -1=Opponent, 0=Empty
            # Player 1 turn: 1=Black, 2=White. 
            # Player 2 turn: 1=White, 2=Black (flip perspective)
            
            p = engine.current_player
            board_input = np.zeros((15, 15), dtype=np.float32)
            
            for r in range(15):
                for c in range(15):
                    v = engine.board.board[r][c]
                    if v == 0: board_input[r][c] = 0
                    elif v == p: board_input[r][c] = 1.0  # Me
                    else: board_input[r][c] = -1.0        # Opponent
            
            # Get Teacher Move
            # Uses AlphaBeta
            move = teacher.get_move(engine.board, p)
            mx, my = move
            
            if mx == -1: # Resign or Error
                break
                
            # Store tuple (Input, OutputClass)
            # OutputClass = 15*x + y
            action_idx = mx * 15 + my
            dataset.append((board_input, action_idx))
            
            engine.make_move(mx, my)
            game_moves += 1
            
        total_moves += game_moves
        print(f"Game {i+1}/{num_games} finished ({game_moves} moves). Total samples: {len(dataset)}")

    print(f"Generation complete. {len(dataset)} samples collected in {time.time()-start_time:.1f}s.")
    
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    # Generate 300 games (~15,000 samples)
    # This provides enough data for the model to beat Random/Greedy
    generate_dataset(num_games=300, save_path="data/sl_dataset_v1.pkl")
