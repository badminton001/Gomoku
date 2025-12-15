"""数据分析脚本 - 处理五线程评估结果并生成统计数据

从五线程评估的原始结果生成可视化所需的数据文件：
- preprocessed_data.csv
- win_rates.csv
- response_times.csv
- matchup_matrix.csv
- elo_ratings.csv
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple


def find_latest_result_file(results_dir: str) -> Path:
    """找到最新的评估结果文件"""
    results_path = Path(results_dir)
    
    # 查找CSV文件
    csv_files = list(results_path.glob("results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"没有在 {results_dir} 中找到评估结果文件")
    
    # 返回最新的文件
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个AI的胜率统计"""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    win_rates_data = []
    
    for ai in ai_names:
        # 作为P1的记录
        df_as_p1 = df[df['player1'] == ai]
        # 作为P2的记录
        df_as_p2 = df[df['player2'] == ai]
        
        # 计算胜利
        p1_wins = len(df_as_p1[df_as_p1['winner'] == 'player1'])
        p2_wins = len(df_as_p2[df_as_p2['winner'] == 'player2'])
        total_wins = p1_wins + p2_wins
        
        # 计算失败
        p1_losses = len(df_as_p1[df_as_p1['winner'] == 'player2'])
        p2_losses = len(df_as_p2[df_as_p2['winner'] == 'player1'])
        total_losses = p1_losses + p2_losses
        
        # 计算平局
        p1_draws = len(df_as_p1[df_as_p1['winner'] == 'draw'])
        p2_draws = len(df_as_p2[df_as_p2['winner'] == 'draw'])
        total_draws = p1_draws + p2_draws
        
        # 总游戏数
        total_games = len(df_as_p1) + len(df_as_p2)
        
        # 胜率
        win_rate = total_wins / total_games if total_games > 0 else 0
        
        win_rates_data.append({
            'algorithm': ai,
            'wins': total_wins,
            'losses': total_losses,
            'draws': total_draws,
            'total_games': total_games,
            'win_rate': win_rate
        })
    
    win_rates_df = pd.DataFrame(win_rates_data)
    win_rates_df = win_rates_df.sort_values('win_rate', ascending=False).reset_index(drop=True)
    
    return win_rates_df


def calculate_response_times(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个AI的响应时间统计"""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    time_stats_data = []
    
    for ai in ai_names:
        # 收集该AI的所有响应时间
        as_p1_times = df[df['player1'] == ai]['player1_avg_time']
        as_p2_times = df[df['player2'] == ai]['player2_avg_time']
        all_times = pd.concat([as_p1_times, as_p2_times])
        
        time_stats_data.append({
            'algorithm': ai,
            'mean_time': all_times.mean(),
            'median_time': all_times.median(),
            'std_time': all_times.std(),
            'min_time': all_times.min(),
            'max_time': all_times.max()
        })
    
    time_stats_df = pd.DataFrame(time_stats_data)
    time_stats_df = time_stats_df.sort_values('mean_time').reset_index(drop=True)
    
    return time_stats_df


def calculate_matchup_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算配对矩阵（行对列的胜率）"""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    # 创建矩阵
    matrix = pd.DataFrame(0.0, index=ai_names, columns=ai_names)
    
    for ai1 in ai_names:
        for ai2 in ai_names:
            if ai1 == ai2:
                matrix.loc[ai1, ai2] = 0.5  # 自己对自己
                continue
            
            # ai1 作为 P1 对 ai2 作为 P2
            games_p1 = df[(df['player1'] == ai1) & (df['player2'] == ai2)]
            wins_p1 = len(games_p1[games_p1['winner'] == 'player1'])
            
            # ai1 作为 P2 对 ai2 作为 P1
            games_p2 = df[(df['player1'] == ai2) & (df['player2'] == ai1)]
            wins_p2 = len(games_p2[games_p2['winner'] == 'player2'])
            
            total_games = len(games_p1) + len(games_p2)
            total_wins = wins_p1 + wins_p2
            
            win_rate = total_wins / total_games if total_games > 0 else 0
            matrix.loc[ai1, ai2] = win_rate
    
    return matrix


def calculate_elo_ratings(df: pd.DataFrame, k_factor: int = 32, initial_rating: int = 1500) -> pd.DataFrame:
    """使用ELO评分系统计算算法评分"""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    # 初始化ELO评分
    elo_ratings = {ai: initial_rating for ai in ai_names}
    
    # 遍历每场游戏更新ELO
    for _, row in df.iterrows():
        p1, p2 = row['player1'], row['player2']
        winner = row['winner']
        
        # 当前评分
        r1, r2 = elo_ratings[p1], elo_ratings[p2]
        
        # 期望胜率
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # 实际得分
        if winner == 'player1':
            s1, s2 = 1, 0
        elif winner == 'player2':
            s1, s2 = 0, 1
        else:  # draw
            s1, s2 = 0.5, 0.5
        
        # 更新评分
        elo_ratings[p1] = r1 + k_factor * (s1 - e1)
        elo_ratings[p2] = r2 + k_factor * (s2 - e2)
    
    # 转换为DataFrame
    elo_df = pd.DataFrame([
        {'algorithm': ai, 'elo_rating': rating}
        for ai, rating in elo_ratings.items()
    ])
    elo_df = elo_df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
    
    return elo_df


def add_game_categories(df: pd.DataFrame) -> pd.DataFrame:
    """添加游戏分类"""
    df = df.copy()
    
    # 游戏长度分类
    def categorize_length(moves):
        if moves < 50:
            return 'Short'
        elif moves < 100:
            return 'Medium'
        else:
            return 'Long'
    
    df['game_length_category'] = df['total_moves'].apply(categorize_length)
    
    return df


def analyze_results(input_file: Path, output_dir: Path):
    """分析评估结果并生成所有统计数据"""
    print("=" * 80)
    print(" 数据分析")
    print("=" * 80)
    
    # 加载原始数据
    print(f"\n📂 加载数据: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   共 {len(df)} 局游戏")
    
    # 添加分类
    df = add_game_categories(df)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存预处理数据
    preprocessed_path = output_dir / "preprocessed_data.csv"
    df.to_csv(preprocessed_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 预处理数据: {preprocessed_path}")
    
    # 2. 计算胜率
    print("\n计算胜率统计...")
    win_rates = calculate_win_rates(df)
    win_rates_path = output_dir / "win_rates.csv"
    win_rates.to_csv(win_rates_path, index=False, encoding='utf-8-sig')
    print(f"✓ 胜率统计: {win_rates_path}")
    
    # 3. 计算响应时间
    print("\n计算响应时间统计...")
    time_stats = calculate_response_times(df)
    time_stats_path = output_dir / "response_times.csv"
    time_stats.to_csv(time_stats_path, index=False, encoding='utf-8-sig')
    print(f"✓ 响应时间: {time_stats_path}")
    
    # 4. 计算配对矩阵
    print("\n计算配对矩阵...")
    matchup_matrix = calculate_matchup_matrix(df)
    matchup_path = output_dir / "matchup_matrix.csv"
    matchup_matrix.to_csv(matchup_path, encoding='utf-8-sig')
    print(f"✓ 配对矩阵: {matchup_path}")
    
    # 5. 计算ELO评分
    print("\n计算ELO评分...")
    elo_ratings = calculate_elo_ratings(df)
    elo_path = output_dir / "elo_ratings.csv"
    elo_ratings.to_csv(elo_path, index=False, encoding='utf-8-sig')
    print(f"✓ ELO评分: {elo_path}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print(" 分析完成")
    print("=" * 80)
    
    print("\n📊 胜率排行:")
    for idx, row in win_rates.iterrows():
        print(f"  {idx+1}. {row['algorithm']:15s}: {row['win_rate']:.1%} ({row['wins']}/{row['total_games']})")
    
    print("\n⚡ 响应时间:")
    for idx, row in time_stats.iterrows():
        print(f"  {row['algorithm']:15s}: {row['mean_time']:.3f}s (中位数: {row['median_time']:.3f}s)")
    
    print("\n🏆 ELO评分:")
    for idx, row in elo_ratings.iterrows():
        print(f"  {idx+1}. {row['algorithm']:15s}: {row['elo_rating']:.0f}")
    
    print(f"\n✅ 所有分析数据已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分析五线程评估结果")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data/results/self_play/5threads',
        help='评估结果目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/results',
        help='输出目录'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='直接指定输入CSV文件（覆盖--input-dir）'
    )
    
    args = parser.parse_args()
    
    # 确定输入文件
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        input_file = find_latest_result_file(args.input_dir)
    
    output_dir = Path(args.output_dir)
    
    # 执行分析
    analyze_results(input_file, output_dir)


if __name__ == "__main__":
    main()
