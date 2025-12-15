import optuna
import random
import logging
import sys
import os

# Add root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine.game_engine import GameEngine
from backend.ai.basic.strong_ai import AlphaBetaAgent
from backend.ai.basic.classic_ai import GreedyAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

print("Starting AI Auto-Tuning System (Optuna)...")

    # Define hyperparameter search space
    # 1. Search Depth (Expensive)
    search_depth = trial.suggest_int('search_depth', 2, 4)
    # 2. Time Limit (Latency vs Accuracy)
    time_limit = trial.suggest_float('time_limit', 0.5, 3.0)
    
    logger.info(f"--- Trial {trial.number}: Depth={search_depth}, Time={time_limit:.1f}s ---")

    # Instantiate Agents
    try:
        candidate_ai = AlphaBetaAgent(depth=search_depth, time_limit=time_limit)
        opponent_ai = GreedyAgent(distance=2) # Baseline Opponent
        
        # Run Evaluation Matches (Real Simulation)
        # We run 2 games: 1 as Black, 1 as White
        total_score = 0
        games = 2
        
        for i in range(games):
            engine = GameEngine(board_size=15)
            # Game 1: Candidate is Player 1 (Black)
            if i == 0:
                p1, p2 = candidate_ai, opponent_ai
                cand_color = 1
            else:
                p1, p2 = opponent_ai, candidate_ai
                cand_color = 2
                
            # Play loop
            max_moves = 60 # Fast evaluation
            while not engine.game_over and engine.board.move_count < max_moves:
                curr_player = engine.current_player
                if curr_player == 1:
                    m = p1.get_move(engine.board, 1)
                else:
                    m = p2.get_move(engine.board, 2)
                    
                if m == (-1, -1):
                    engine.game_over = True
                    engine.winner = 3 - curr_player
                else:
                    engine.make_move(m[0], m[1])
            
            # Scoring
            if engine.winner == cand_color:
                total_score += 1.0 # Win
            elif engine.winner == 0:
                total_score += 0.5 # Draw (or timeout limit reached)
            else:
                total_score -= 1.0 # Loss
                
            # Bonus for speed (if won)
            # This encourages efficient wins
            if engine.winner == cand_color:
                 total_score += 0.1 * (max_moves - engine.board.move_count) / max_moves

        avg_score = total_score / games
        logger.info(f"   -> Result Score: {avg_score:.2f}")
        return avg_score

    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return -100.0 # Heavy penalty for crash

if __name__ == "__main__":
    # Create a study object and optimize the objective function
    # Pruning allows stopping unpromising trials early
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    print("Executing 10 automated tuning trials (Real Matches)...")
    # Reduced to 10 trials for demonstration speed, but real tuning would be 50+
    study.optimize(objective, n_trials=10)

    # Output results
    print("\n" + "="*30)
    print("Tuning Completed! Best results:")
    print(f"Best Score: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    print("="*30)