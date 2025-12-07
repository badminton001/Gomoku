"""模型评估脚本 (Person E)

运行自对弈评估，收集性能数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.winplay_service import SelfPlayEngine
from backend.algorithms.classic_ai import GreedyAgent, MinimaxAgent, AlphaBetaAgent
# from backend.algorithms.learning_ai import DQNAgent, PPOAgent  # 待实现


def main():
    """运行模型评估"""
    print("=" * 60)
    print(" Model Evaluation - Self-Play Tournament")
    print("=" * 60)
    
    # 初始化引擎
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # 注册AI算法（跳过MCTS和AlphaBeta-D3优化速度）
    print("\n📋 Registering AI algorithms...")
    engine.register_ai("Greedy", GreedyAgent(distance=2))
    engine.register_ai("Minimax-D2", MinimaxAgent(depth=2, distance=2, candidate_limit=10))
    engine.register_ai("AlphaBeta-D2", AlphaBetaAgent(depth=2, distance=2, candidate_limit=10))
    # engine.register_ai("AlphaBeta-D3", AlphaBetaAgent(depth=3, distance=2, candidate_limit=12))  # 太慢，已移除
    
    # TODO: 添加强化学习算法
    # engine.register_ai("DQN", DQNAgent.load("./models/dqn_best.pkl"))
    # engine.register_ai("PPO", PPOAgent.load("./models/ppo_best.pkl"))
    
    print(f"\n✓ Registered {len(engine.ai_algorithms)} AI algorithms\n")
    
    # 运行循环赛
    num_games = 20  # 每对AI对战20次
    print(f"🎮 Running tournament ({num_games} games per pair)...\n")
    
    results = engine.run_round_robin(num_games_per_pair=num_games, verbose=True)
    
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
