"""Single Match Visualization Tool

Watch a single AI vs AI match with detailed move-by-move output
Perfect for debugging slow algorithms like MCTS
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.winplay_service import SelfPlayEngine
from backend.algorithms.classic_ai import GreedyAgent, MinimaxAgent, AlphaBetaAgent
from backend.algorithms.mcts_ai import MCTSAgent


def print_board(board):
    """Print current board state"""
    print("\n   " + "  ".join([f"{i:2d}" for i in range(board.size)]))
    for i in range(board.size):
        row = f"{i:2d} "
        for j in range(board.size):
            cell = board.board[i][j]
            if cell == 0:
                row += " · "
            elif cell == 1:
                row += " ⚫"
            else:
                row += " ⚪"
        print(row)


def watch_single_match(ai1_name, ai1, ai2_name, ai2):
    """Watch a single match with visualization"""
    from backend.models.board import Board
    import time
    
    board = Board(15)
    current_player = 1
    move_count = 0
    
    print("\n" + "="*60)
    print(f" WATCHING: {ai1_name} (⚫) vs {ai2_name} (⚪)")
    print("="*60)
    
    while move_count < 225:  # 15×15
        current_ai = ai1 if current_player == 1 else ai2
        current_name = ai1_name if current_player == 1 else ai2_name
        symbol = "⚫" if current_player == 1 else "⚪"
        
        print(f"\n{'='*60}")
        print(f"Move {move_count + 1}: {current_name} {symbol} thinking...")
        print("="*60)
        
        # Show current board
        print_board(board)
        
        # Get move with timing
        print(f"\n⏱️  Waiting for {current_name} to move...", flush=True)
        start = time.time()
        
        try:
            move = current_ai.get_move(board, current_player)
            elapsed = time.time() - start
            
            if move is None:
                print("\n❌ No valid move found - DRAW")
                break
            
            x, y = move
            
            if not board.is_valid_move(x, y):
                print(f"\n❌ Invalid move ({x},{y}) - {current_name} LOSES")
                break
            
            # Make move
            board.place_stone(x, y, current_player)
            print(f"\n✅ {current_name} played ({x},{y}) in {elapsed:.2f}s")
            
            # Check game result
            result = board.get_game_result()
            if result == current_player:
                print_board(board)
                print(f"\n🎉 {current_name} WINS!")
                break
            elif result == -1:
                print_board(board)
                print("\n🤝 DRAW - Board full")
                break
            
            # Next player
            current_player = 3 - current_player
            move_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Match interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*60)
    print(" MATCH COMPLETE")
    print("="*60)


def main():
    """Run visualization demo"""
    print("="*60)
    print(" SELF-PLAY VISUALIZATION DEMO")
    print("="*60)
    
    print("\nSelect match to watch:")
    print("1. AlphaBeta-D2 vs Greedy (Fast, ~10 seconds)")
    print("2. AlphaBeta-D2 vs MCTS-100 (Slow, ~20+ minutes)")
    print("3. MCTS-100 vs MCTS-100 (Very slow, ~40+ minutes)")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == "1":
        ai1_name = "AlphaBeta-D2"
        ai1 = AlphaBetaAgent(depth=2, distance=2, candidate_limit=10)
        ai2_name = "Greedy"
        ai2 = GreedyAgent(distance=2)
    elif choice == "2":
        ai1_name = "AlphaBeta-D2"
        ai1 = AlphaBetaAgent(depth=2, distance=2, candidate_limit=10)
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
        ai1 = AlphaBetaAgent(depth=2, distance=2, candidate_limit=10)
        ai2_name = "Greedy"
        ai2 = GreedyAgent(distance=2)
    
    watch_single_match(ai1_name, ai1, ai2_name, ai2)


if __name__ == "__main__":
    main()
