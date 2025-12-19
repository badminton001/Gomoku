"""Match Visualization Tool"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.services.winplay_service import SelfPlayEngine
from backend.ai.baselines import GreedyAgent
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.mcts import MCTSAgent


def print_board(board):
    """Print board."""
    print("\n   " + "  ".join([f"{i:2d}" for i in range(board.size)]))
    for i in range(board.size):
        row = f"{i:2d} "
        for j in range(board.size):
            cell = board.board[i][j]
            if cell == 0:
                row += " . "
            elif cell == 1:
                row += " X "
            else:
                row += " O "
        print(row)


def watch_single_match(ai1_name, ai1, ai2_name, ai2):
    """Watch match."""
    from backend.engine.board import Board
    import time
    
    board = Board(15)
    current_player = 1
    move_count = 0
    
    print(f"\n[WATCHING] {ai1_name} (X) vs {ai2_name} (O)")
    
    while move_count < 225:  # 15x15
        current_ai = ai1 if current_player == 1 else ai2
        current_name = ai1_name if current_player == 1 else ai2_name
        symbol = "X" if current_player == 1 else "O"
        
        print(f"\nMove {move_count + 1}: {current_name} {symbol} thinking...")
        
        # Show board
        print_board(board)
        
        # Get move
        print(f"\n[WAIT] Waiting for {current_name} to move...", flush=True)
        start = time.time()
        
        try:
            move = current_ai.get_move(board, current_player)
            elapsed = time.time() - start
            
            if move is None:
                print("\n[END] No valid move found - DRAW")
                break
            
            x, y = move
            
            if not board.is_valid_move(x, y):
                print(f"\n[ERROR] Invalid move ({x},{y}) - {current_name} LOSES")
                break
            
            # Move
            board.place_stone(x, y, current_player)
            print(f"\n[MOVE] {current_name} played ({x},{y}) in {elapsed:.2f}s")
            
            # Check result
            result = board.get_game_result()
            if result == current_player:
                print_board(board)
                print(f"\n[WIN] {current_name} WINS!")
                break
            elif result == -1:
                print_board(board)
                print("\n[DRAW] Board full")
                break
            
            # Next player
            current_player = 3 - current_player
            move_count += 1
            
        except KeyboardInterrupt:
            print("\n\n[WARN] Match interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n[MATCH COMPLETE]")


def main():
    """Run demo."""
    print(" SELF-PLAY VISUALIZATION DEMO")
    
    print("\nSelect match to watch:")
    print("1. AlphaBeta-D2 vs Greedy (Fast, ~10 seconds)")
    print("2. AlphaBeta-D2 vs MCTS-100 (Slow, ~20+ minutes)")
    print("3. MCTS-100 vs MCTS-100 (Very slow, ~40+ minutes)")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == "1":
        ai1_name = "AlphaBeta-D2"
        ai1 = AlphaBetaAgent(depth=2, time_limit=2.0)
        ai2_name = "Greedy"
        ai2 = GreedyAgent(distance=2)
    elif choice == "2":
        ai1_name = "AlphaBeta-D2"
        ai1 = AlphaBetaAgent(depth=2, time_limit=2.0)
        ai2_name = "MCTS-100"
        ai2 = MCTSAgent(iteration_limit=100)
    elif choice == "3":
        ai1_name = "MCTS-100"
        ai1 = MCTSAgent(iteration_limit=100)
        ai2_name = "MCTS-100-2"
        ai2 = MCTSAgent(iteration_limit=100)
    else:
        print("Invalid choice, defaulting to option 1")
        ai1_name = "AlphaBeta-D2"
        ai1 = AlphaBetaAgent(depth=2, time_limit=2.0)
        ai2_name = "Greedy"
        ai2 = GreedyAgent(distance=2)
    
    watch_single_match(ai1_name, ai1, ai2_name, ai2)


if __name__ == "__main__":
    main()
