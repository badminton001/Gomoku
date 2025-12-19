"""Model Evaluation Script"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.api.services.winplay_service import SelfPlayEngine
from backend.ai.baselines import GreedyAgent
from backend.ai.minimax import AlphaBetaAgent
from backend.ai.mcts import MCTSAgent
from pathlib import Path


def check_checkpoint():
    """Check/Resume Checkpoint."""
    checkpoint_path = Path("./data/results/self_play/checkpoint.json")
    if checkpoint_path.exists():
        print(f"\n[WARN] CHECKPOINT DETECTED")
        print("\n[INFO] Found an incomplete evaluation session from a previous run.")
        print("   Checkpoint file: ./data/results/self_play/checkpoint.json")
        
        response = input("\n[?] Do you want to resume from the checkpoint? (y/n): ").strip().lower()
        return response == 'y'
    return False


def main():
    """Run all algorithms."""
    print("Model Evaluation - Complete AI Tournament")
    
    # Check checkpoint
    resume_from_checkpoint = check_checkpoint()
    
    # Init engine
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # Register AIs
    print("\n[INFO] Registering AI algorithms...")
    
    # Classic
    engine.register_ai("Greedy", GreedyAgent(distance=2))
    engine.register_ai("Minimax-D2", AlphaBetaAgent(depth=2, time_limit=2.0)) # Maps to AlphaBeta as Minimax
    engine.register_ai("AlphaBeta-D2", AlphaBetaAgent(depth=2, time_limit=2.0))
    
    # MCTS
    engine.register_ai("MCTS-100", MCTSAgent(iteration_limit=100))
    
    # DQN (if exists)
    try:
        from backend.ai.advanced.qlearning_ai import QLearningAgent
        dqn_agent = QLearningAgent(model_path="models/dqn_v1_final")
        engine.register_ai("DQN", dqn_agent)
        if dqn_agent.model:
            print("   [OK] DQN model loaded: models/dqn_v1_final.zip")
        else:
            print("   [WARN] DQN: No model found, will use random moves")
    except Exception as e:
        print(f"   [WARN] DQN not available: {e}")
    
    print(f"\n[OK] Registered {len(engine.ai_algorithms)} AI algorithms")
    print(f"\n[OK] Registered {len(engine.ai_algorithms)} AI algorithms")
    
    # Info
    print("\n[INFO] Tournament Configuration:")
    print(f"   • Algorithms: {', '.join(engine.ai_algorithms.keys())}")
    
    # Config
    num_games = 10  # 10 games per pair (balanced for time)
    total_matches = len(engine.ai_algorithms) * (len(engine.ai_algorithms) - 1) * num_games
    
    print(f"   • Games per pair: {num_games}")
    print(f"   • Total matches: {total_matches}")
    print(f"\n[OK] MCTS performance Optimized - expected completion: 4-6 hours")
    print(f"   (MCTS now skips forbidden-move checks during candidate generation)")
    
    # Run
    print(f"\n[START] Starting tournament...\n")
    
    results = engine.run_round_robin(num_games_per_pair=num_games, verbose=True, resume=resume_from_checkpoint)
    
    # Save
    print("\n[SAVE] Saving results...")
    engine.save_results(results, output_dir='./data/results/self_play')
    
    # Cleanup
    engine.cleanup()
    
    print("\nEvaluation Complete!")
    print(f"\n[OK] Total games: {len(results)}")
    print(f"[INFO] Results saved to: ./data/results/self_play/")
    print(f"\n[NEXT] Next steps:")
    print(f"   1. python scripts/analyze_performance.py")
    print(f"   2. python scripts/generate_visualizations.py")
    print(f"   3. python scripts/generate_reports.py")


if __name__ == "__main__":
    main()
