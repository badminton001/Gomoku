"""Model Evaluation Script (Person E)

Run self-play evaluation and collect performance data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.winplay_service import SelfPlayEngine
from backend.algorithms.classic_ai import GreedyAgent, MinimaxAgent, AlphaBetaAgent
from backend.algorithms.mcts_ai import MCTSAgent
# from backend.algorithms.learning_ai import DQNAgent, PPOAgent  # 待实现
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
    """运行模型评估"""
    print("=" * 60)
    print(" Model Evaluation - Self-Play Tournament")
    print("=" * 60)
    
    # Check for checkpoint
    resume_from_checkpoint = check_checkpoint()
    
    # Initialize engine
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # Register AI algorithms (including MCTS)
    print("\n📋 Registering AI algorithms...")
    engine.register_ai("Greedy", GreedyAgent(distance=2))
    engine.register_ai("Minimax-D2", MinimaxAgent(depth=2, distance=2, candidate_limit=10))
    engine.register_ai("AlphaBeta-D2", AlphaBetaAgent(depth=2, distance=2, candidate_limit=10))
    engine.register_ai("MCTS-1000", MCTSAgent(iteration_limit=1000))  # Fixed: only iteration_limit
    
    # TODO: Add reinforcement learning algorithms
    # engine.register_ai("DQN", DQNAgent.load("./models/dqn_best.pkl"))
    # engine.register_ai("PPO", PPOAgent.load("./models/ppo_best.pkl"))
    
    print(f"\n✓ Registered {len(engine.ai_algorithms)} AI algorithms\n")
    
    # Run tournament
    num_games = 20  # 20 games per pair
    print(f"🎮 Running tournament ({num_games} games per pair)...\n")
    
    results = engine.run_round_robin(num_games_per_pair=num_games, verbose=True, resume=resume_from_checkpoint)
    
    # 保存结果
    print("\n💾 Saving results...")
    engine.save_results(results, output_dir='./data/results/self_play')
    
    # 清理
    engine.cleanup()
    
    print("\n" + "=" * 60)
    print(" Evaluation Complete!")
    print("=" * 60)
    print(f"\n✅ Total games: {len(results)}")
    print(f"📁 Results saved to: ./data/results/self_play/")
    print(f"\n🎯 Next step: Run scripts/analyze_performance.py to analyze data")


if __name__ == "__main__":
    main()
