"""分布式自对弈系统 - 支持并行运行和结果合并

使用方法：
1. 独立运行多个小批次：
   python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 50

2. 合并所有批次结果：
   python scripts/parallel_eval.py --merge

这样可以在多台机器或多个进程中并行运行，大大加快评估速度。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd

from backend.services.winplay_service import SelfPlayEngine, GameResult
from backend.algorithms.classic_ai import GreedyAgent, MinimaxAgent, AlphaBetaAgent
from backend.algorithms.mcts_ai import MCTSAgent
from backend.algorithms.qlearning_ai import QLearningAgent


def get_batch_output_path(batch_id: int) -> Path:
    """获取批次输出路径"""
    return Path(f"./data/results/self_play/batch_{batch_id}.json")


def get_merge_output_dir() -> Path:
    """获取合并输出目录"""
    return Path("./data/results/self_play/merged")


def run_batch(batch_id: int, total_batches: int, games_per_batch: int, 
              algorithms: List[str] = None):
    """运行一个批次的自对弈
    
    Args:
        batch_id: 批次ID (1-based)
        total_batches: 总批次数
        games_per_batch: 每个批次的游戏数
        algorithms: 要测试的算法列表
    """
    print("=" * 70)
    print(f" 批次 {batch_id}/{total_batches} - 自对弈评估")
    print("=" * 70)
    print(f"每个配对: {games_per_batch} 局")
    
    # 初始化引擎
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # 注册AI算法
    if algorithms is None:
        algorithms = ["Greedy", "Minimax-D2", "AlphaBeta-D2", "MCTS-300", "DQN"]
    
    print("\n注册AI算法...")
    if "Greedy" in algorithms:
        engine.register_ai("Greedy", GreedyAgent(distance=2))
    if "Minimax-D2" in algorithms:
        engine.register_ai("Minimax-D2", MinimaxAgent(depth=2, distance=2, candidate_limit=10))
    if "AlphaBeta-D2" in algorithms:
        engine.register_ai("AlphaBeta-D2", AlphaBetaAgent(depth=2, distance=2, candidate_limit=10))
    if "MCTS-300" in algorithms:
        engine.register_ai("MCTS-300", MCTSAgent(iteration_limit=300))
    if "DQN" in algorithms:
        try:
            dqn_agent = QLearningAgent(model_path="models/dqn_15x15_final")
            engine.register_ai("DQN", dqn_agent)
        except Exception as e:
            print(f"   ⚠️ DQN not available: {e}")
    
    print(f"\n✓ 注册了 {len(engine.ai_algorithms)} 个AI")
    
    # 运行循环赛
    print(f"\n开始批次 {batch_id} 的对局...")
    results = engine.run_round_robin(
        num_games_per_pair=games_per_batch, 
        verbose=True,
        resume=False
    )
    
    # 保存批次结果
    output_path = get_batch_output_path(batch_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    batch_data = {
        'batch_id': batch_id,
        'total_batches': total_batches,
        'games_per_batch': games_per_batch,
        'timestamp': datetime.now().isoformat(),
        'algorithms': list(engine.ai_algorithms.keys()),
        'results': [r.to_dict() for r in results]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 批次 {batch_id} 完成!")
    print(f"   共 {len(results)} 局游戏")
    print(f"   结果保存到: {output_path}")
    
    engine.cleanup()
    return results


def merge_batches():
    """合并所有批次的结果"""
    print("=" * 70)
    print(" 合并批次结果")
    print("=" * 70)
    
    # 查找所有批次文件
    batch_files = sorted(Path("./data/results/self_play").glob("batch_*.json"))
    
    if not batch_files:
        print("\n❌ 没有找到任何批次文件!")
        print("   请先运行批次: python scripts/parallel_eval.py --batch-id 1")
        return
    
    print(f"\n找到 {len(batch_files)} 个批次文件:")
    for bf in batch_files:
        print(f"  - {bf.name}")
    
    # 加载并合并所有结果
    all_results = []
    batch_info = []
    
    for batch_file in batch_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        batch_info.append({
            'batch_id': batch_data['batch_id'],
            'games': len(batch_data['results']),
            'timestamp': batch_data['timestamp']
        })
        
        all_results.extend(batch_data['results'])
    
    print(f"\n合并统计:")
    print(f"  总批次数: {len(batch_files)}")
    print(f"  总游戏数: {len(all_results)}")
    
    # 创建输出目录
    output_dir = get_merge_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存合并后的详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_json = output_dir / f"merged_results_{timestamp}.json"
    
    merged_data = {
        'merge_timestamp': datetime.now().isoformat(),
        'total_games': len(all_results),
        'batches': batch_info,
        'results': all_results
    }
    
    with open(merged_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细结果: {merged_json}")
    
    # 保存CSV格式
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"merged_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV结果: {csv_path}")
    
    # 生成汇总统计
    print(f"\n📊 汇总统计:")
    print(f"   平均步数: {df['total_moves'].mean():.1f}")
    print(f"   平均时间: {(df['player1_avg_time'] + df['player2_avg_time']).mean() / 2:.3f}s")
    
    # 胜率统计
    print(f"\n🏆 胜率统计:")
    ai_names = set(df['player1'].unique()) | set(df['player2'].unique())
    for ai in sorted(ai_names):
        df_as_p1 = df[df['player1'] == ai]
        df_as_p2 = df[df['player2'] == ai]
        
        p1_wins = len(df_as_p1[df_as_p1['winner'] == 'player1'])
        p2_wins = len(df_as_p2[df_as_p2['winner'] == 'player2'])
        
        total_games = len(df_as_p1) + len(df_as_p2)
        total_wins = p1_wins + p2_wins
        
        if total_games > 0:
            win_rate = total_wins / total_games * 100
            print(f"   {ai:20s}: {total_wins:3d}/{total_games:3d} = {win_rate:5.1f}%")
    
    print("\n" + "=" * 70)
    print(" 合并完成!")
    print("=" * 70)
    print(f"\n📁 结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="分布式自对弈评估系统")
    
    # 批次模式
    parser.add_argument('--batch-id', type=int, help='批次ID (1-based)')
    parser.add_argument('--total-batches', type=int, default=4, help='总批次数')
    parser.add_argument('--games-per-batch', type=int, default=5, 
                       help='每个配对在每个批次中的游戏数')
    
    # 合并模式
    parser.add_argument('--merge', action='store_true', help='合并所有批次结果')
    
    # 算法选择
    parser.add_argument('--algorithms', nargs='+', 
                       default=["Greedy", "Minimax-D2", "AlphaBeta-D2", "MCTS-300", "DQN"],
                       help='要测试的算法')
    
    args = parser.parse_args()
    
    if args.merge:
        # 合并模式
        merge_batches()
    elif args.batch_id:
        # 批次模式
        run_batch(args.batch_id, args.total_batches, args.games_per_batch, args.algorithms)
    else:
        # 显示帮助
        parser.print_help()
        print("\n示例用法:")
        print("  # 运行批次1 (共4个批次)")
        print("  python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5")
        print("\n  # 运行批次2")
        print("  python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5")
        print("\n  # 合并所有批次")
        print("  python scripts/parallel_eval.py --merge")


if __name__ == "__main__":
    main()
