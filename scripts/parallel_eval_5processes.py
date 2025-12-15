"""五进程并行自对弈评估系统 - 真正的并行执行

使用multiprocessing实现真正的5进程同时运行，突破Python GIL限制。

使用方法：
    python scripts/parallel_eval_5processes.py --games-per-pair 17
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd

from backend.api.services.winplay_service import SelfPlayEngine, GameResult
from backend.ai.basic.classic_ai import GreedyAgent
from backend.ai.basic.strong_ai import AlphaBetaAgent
from backend.ai.advanced.mcts_ai import MCTSAgent
from backend.ai.advanced.qlearning_ai import QLearningAgent
from backend.ai.advanced.hybrid_ai import HybridAgent


def get_output_dir() -> Path:
    """获取输出目录"""
    return Path("./data/results/self_play/5processes")


def create_ai_agents() -> Dict[str, any]:
    """创建所有AI代理实例"""
    agents = {}
    agents["Greedy"] = GreedyAgent(distance=2)
    agents["AlphaBeta"] = AlphaBetaAgent(depth=4, time_limit=4.0)
    agents["Minimax"] = AlphaBetaAgent(depth=2, time_limit=2.0)
    agents["MCTS-500"] = MCTSAgent(iteration_limit=500)
    
    try:
        agents["DQN"] = QLearningAgent(model_path="models/dqn_15x15_final")
    except Exception as e:
        print(f"[警告] DQN加载失败: {e}")
        agents["DQN"] = None
    
    try:
        agents["Hybrid"] = HybridAgent(model_path="models/sl_model_v1.pth", device="cpu")
    except Exception as e:
        print(f"[警告] Hybrid加载失败: {e}")
        agents["Hybrid"] = None
    
    agents = {k: v for k, v in agents.items() if v is not None}
    return agents


def get_matchups(ai_names: List[str]) -> List[Tuple[str, str]]:
    """生成所有配对组合"""
    matchups = []
    for i, ai1 in enumerate(ai_names):
        for ai2 in ai_names[i+1:]:
            matchups.append((ai1, ai2))
    return matchups


def run_process_batch(process_id: int, matchups: List[Tuple[str, str]], games_per_pair: int):
    """在独立进程中运行一批配对
    
    Args:
        process_id: 进程ID
        matchups: 该进程要处理的配对列表
        games_per_pair: 每个配对的游戏数（单向）
    """
    print(f"[进程 {process_id}] 🚀 启动，处理 {len(matchups)} 个配对")
    
    # 每个进程独立创建AI代理
    ai_agents = create_ai_agents()
    print(f"[进程 {process_id}] ✓ AI代理创建完成: {sorted(ai_agents.keys())}")
    
    # 创建引擎
    engine = SelfPlayEngine(board_size=15, use_wandb=False)
    
    # 注册需要的AI
    registered_ais = set()
    for ai1, ai2 in matchups:
        if ai1 not in registered_ais:
            engine.register_ai(ai1, ai_agents[ai1])
            registered_ais.add(ai1)
        if ai2 not in registered_ais:
            engine.register_ai(ai2, ai_agents[ai2])
            registered_ais.add(ai2)
    
    print(f"[进程 {process_id}] ✓ 已注册 {len(registered_ais)} 个AI")
    
    # 运行每个配对
    batch_results = []
    total_games = len(matchups) * games_per_pair * 2
    completed = 0
    
    for idx, (ai1, ai2) in enumerate(matchups, 1):
        print(f"[进程 {process_id}] 📋 配对 {idx}/{len(matchups)}: {ai1} vs {ai2}")
        
        for game_num in range(games_per_pair):
            # P1 vs P2
            try:
                result1 = engine.play_single_match(ai1, ai2, verbose=False)
                batch_results.append(result1.to_dict())
                completed += 1
                print(f"[进程 {process_id}]   ✓ {completed}/{total_games}: {ai1}先手 胜者={result1.winner} {result1.total_moves}步")
            except Exception as e:
                print(f"[进程 {process_id}]   ✗ 失败: {e}")
            
            # P2 vs P1
            try:
                result2 = engine.play_single_match(ai2, ai1, verbose=False)
                batch_results.append(result2.to_dict())
                completed += 1
                print(f"[进程 {process_id}]   ✓ {completed}/{total_games}: {ai2}先手 胜者={result2.winner} {result2.total_moves}步")
            except Exception as e:
                print(f"[进程 {process_id}]   ✗ 失败: {e}")
    
    engine.cleanup()
    
    # 保存该进程的结果
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_file = output_dir / f"process_{process_id}.json"
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump({
            'process_id': process_id,
            'matchups': matchups,
            'games_per_pair': games_per_pair,
            'results': batch_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[进程 {process_id}] 🎉 完成！共 {len(batch_results)} 局，已保存")


def merge_process_results():
    """合并所有进程的结果"""
    output_dir = get_output_dir()
    process_files = sorted(output_dir.glob("process_*.json"))
    
    if not process_files:
        print("❌ 没有找到进程结果文件")
        return
    
    print(f"\n📁 找到 {len(process_files)} 个进程结果文件")
    
    all_results = []
    for pf in process_files:
        with open(pf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.extend(data['results'])
            print(f"  ✓ {pf.name}: {len(data['results'])} 局")
    
    # 保存合并结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON
    merged_json = output_dir / f"merged_results_{timestamp}.json"
    with open(merged_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_games': len(all_results),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ JSON结果: {merged_json}")
    
    # CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / f"merged_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV结果: {csv_path}")
    
    # 统计
    print(f"\n📊 统计摘要:")
    print(f"  总游戏数: {len(all_results)}")
    print(f"  平均步数: {df['total_moves'].mean():.1f}")
    print(f"  平均时间: {(df['player1_avg_time'] + df['player2_avg_time']).mean() / 2:.3f}s")
    
    # 胜率
    print(f"\n🏆 胜率排行:")
    ai_names = set(df['player1'].unique()) | set(df['player2'].unique())
    
    win_rates = []
    for ai in ai_names:
        df_as_p1 = df[df['player1'] == ai]
        df_as_p2 = df[df['player2'] == ai]
        
        p1_wins = len(df_as_p1[df_as_p1['winner'] == 'player1'])
        p2_wins = len(df_as_p2[df_as_p2['winner'] == 'player2'])
        
        total_games = len(df_as_p1) + len(df_as_p2)
        total_wins = p1_wins + p2_wins
        
        if total_games > 0:
            win_rate = total_wins / total_games * 100
            win_rates.append((ai, total_wins, total_games, win_rate))
    
    win_rates.sort(key=lambda x: x[3], reverse=True)
    
    for rank, (ai, wins, games, rate) in enumerate(win_rates, 1):
        print(f"  {rank}. {ai:15s}: {wins:3d}/{games:3d} = {rate:5.1f}%")
    
    # 保存统计
    stats_path = output_dir / f"statistics_{timestamp}.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("五进程并行评估统计报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总游戏数: {len(all_results)}\n\n")
        f.write("胜率统计:\n")
        for rank, (ai, wins, games, rate) in enumerate(win_rates, 1):
            f.write(f"  {rank}. {ai:15s}: {wins:3d}/{games:3d} = {rate:5.1f}%\n")
    
    print(f"✓ 统计报告: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="五进程并行自对弈评估（真正的并行）")
    parser.add_argument('--games-per-pair', type=int, default=17, 
                       help='每个配对的游戏数（单向）')
    parser.add_argument('--merge-only', action='store_true',
                       help='仅合并现有结果，不运行新评估')
    
    args = parser.parse_args()
    
    if args.merge_only:
        merge_process_results()
        return
    
    print("=" * 80)
    print(" 五进程并行自对弈评估系统 (真正的多进程并行)")
    print("=" * 80)
    
    # 创建AI列表
    ai_agents = create_ai_agents()
    ai_names = sorted(ai_agents.keys())
    
    print(f"\n✓ 共 {len(ai_names)} 个AI: {ai_names}")
    
    # 生成配对
    matchups = get_matchups(ai_names)
    total_games = len(matchups) * args.games_per_pair * 2
    
    print(f"\n配对数量: {len(matchups)}")
    print(f"每个配对: {args.games_per_pair} 局 × 2 (轮换先后手)")
    print(f"总游戏数: {total_games}")
    print(f"并行进程: 5")
    
    # 分配任务到5个进程
    num_processes = 5
    batches = [[] for _ in range(num_processes)]
    for i, matchup in enumerate(matchups):
        batches[i % num_processes].append(matchup)
    
    print("\n进程任务分配:")
    for i, batch in enumerate(batches, 1):
        print(f"  进程 {i}: {len(batch)} 个配对 ({len(batch) * args.games_per_pair * 2} 局)")
    
    # 启动5个独立进程
    print("\n" + "=" * 80)
    print(" 启动5个独立进程进行并行评估")
    print("=" * 80 + "\n")
    
    start_time = datetime.now()
    
    processes = []
    for process_id, batch in enumerate(batches, 1):
        if batch:
            p = multiprocessing.Process(
                target=run_process_batch,
                args=(process_id, batch, args.games_per_pair)
            )
            p.start()
            processes.append(p)
            print(f"✓ 进程 {process_id} 已启动 (PID: {p.pid})")
    
    # 等待所有进程完成
    print(f"\n⏳ 等待 {len(processes)} 个进程完成...")
    for p in processes:
        p.join()
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print(" 所有进程已完成")
    print("=" * 80)
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"预计游戏数: {total_games}")
    
    # 合并结果
    print("\n" + "=" * 80)
    print(" 合并进程结果")
    print("=" * 80)
    merge_process_results()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows支持
    main()
