"""Model Evaluation Script (Person E)

Run self-play evaluation and collect performance data
Includes all available algorithms: Classic + MCTS
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.winplay_service import SelfPlayEngine
from backend.algorithms.classic_ai import GreedyAgent, MinimaxAgent, AlphaBetaAgent
from backend.algorithms.mcts_ai import MCTSAgent
from pathlib import Path


def check_checkpoint():
    """Check for existing checkpoint and ask user"""
    checkpoint_path = Path("./data/results/self_play/checkpoint.json")
    if checkpoint_path.exists():
        print("\n" + "=" * 60)
        print(" ⚠️  CHECKPOINT DETECTED")
        print("=" * 60)
        print("\n📌 Found an incomplete evaluation session from a previous run.")
        print("   Checkpoint file: ./data/results/self_play/checkpoint.json")
        
        response = input("\n❓ Do you want to resume from the checkpoint? (y/n): ").strip().lower()
        return response == 'y'
    return False


def main():
    """Run complete model evaluation with all algorithms"""
    print("=" * 60)
    print(" Model Evaluation - Complete AI Tournament")
    print("=" * 60)
    
    # Check for checkpoint
    resume_from_checkpoint = check_checkpoint()
    
    # Initialize engine
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # Register ALL AI algorithms
    print("\n📋 Registering AI algorithms...")
    print("-" * 60)
    
    # Classic algorithms (fast)
    engine.register_ai("Greedy", GreedyAgent(distance=2))
    engine.register_ai("Minimax-D2", MinimaxAgent(depth=2, distance=2, candidate_limit=10))
    engine.register_ai("AlphaBeta-D2", AlphaBetaAgent(depth=2, distance=2, candidate_limit=10))
    
    # MCTS algorithm (slower - optimized with reduced iterations)
    engine.register_ai("MCTS-100", MCTSAgent(iteration_limit=100))
    
    # Q-Learning (DQN) - if model exists
    try:
        from backend.algorithms.qlearning_ai import QLearningAgent
        dqn_agent = QLearningAgent(model_path="models/dqn_15x15_final")
        engine.register_ai("DQN", dqn_agent)
        if dqn_agent.model:
            print("   ✓ DQN model loaded: models/dqn_15x15_final.zip")
        else:
            print("   ⚠️ DQN: No model found, will use random moves")
    except Exception as e:
        print(f"   ⚠️ DQN not available: {e}")
    
    print(f"\n✓ Registered {len(engine.ai_algorithms)} AI algorithms")
    print("-" * 60)
    
    # Display algorithm info
    print("\n📊 Tournament Configuration:")
    print(f"   • Algorithms: {', '.join(engine.ai_algorithms.keys())}")
    
    # Optimized configuration for MCTS inclusion
    num_games = 10  # 10 games per pair (balanced for time)
    total_matches = len(engine.ai_algorithms) * (len(engine.ai_algorithms) - 1) * num_games
    
    print(f"   • Games per pair: {num_games}")
    print(f"   • Total matches: {total_matches}")
    print(f"\n⚠️  Note: MCTS will take longer due to board complexity")
    print(f"   Estimated time: 4-6 hours")
    print("-" * 60)
    
    # Run tournament
    print(f"\n🎮 Starting tournament...\n")
    
    results = engine.run_round_robin(num_games_per_pair=num_games, verbose=True, resume=resume_from_checkpoint)
    
    # Save results
    print("\n💾 Saving results...")
    engine.save_results(results, output_dir='./data/results/self_play')
    
    # Cleanup
    engine.cleanup()
    
    print("\n" + "=" * 60)
    print(" Evaluation Complete!")
    print("=" * 60)
    print(f"\n✅ Total games: {len(results)}")
    print(f"📁 Results saved to: ./data/results/self_play/")
    print(f"\n🎯 Next steps:")
    print(f"   1. python scripts/analyze_performance.py")
    print(f"   2. python scripts/generate_visualizations.py")
    print(f"   3. python scripts/generate_reports.py")


if __name__ == "__main__":
    main()
