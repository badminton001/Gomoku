"""性能分析脚本 (Person E/G)

数据预处理 + 统计分析 + 性能评估
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

from backend.services.performance_analyzer import StatisticalAnalyzer


# ==================== 数据预处理模块 ====================

def preprocess_data(data_path: str) -> pd.DataFrame:
    """数据预处理：清洗、标准化、验证
    
    Args:
        data_path: 原始数据CSV路径
        
    Returns:
        预处理后的DataFrame
    """
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    original_count = len(df)
    
    # 1. 数据清洗
    # 移除异常值（超长对局）
    q99 = df['total_moves'].quantile(0.99)
    df = df[df['total_moves'] <= q99]
    
    # 移除异常耗时（超过60秒）
    df = df[df['player1_avg_time'] < 60]
    df = df[df['player2_avg_time'] < 60]
    
    # 移除缺失值
    df = df.dropna()
    
    # 确保winner字段合法
    valid_winners = ['player1', 'player2', 'draw']
    df = df[df['winner'].isin(valid_winners)]
    
    cleaned_count = len(df)
    removed = original_count - cleaned_count
    print(f"✓ Cleaned: {original_count} -> {cleaned_count} records (removed {removed})")
    
    # 2. 特征工程
    df['total_time'] = df['player1_avg_time'] + df['player2_avg_time']
    df['time_difference'] = np.abs(df['player1_avg_time'] - df['player2_avg_time'])
    df['faster_player'] = np.where(
        df['player1_avg_time'] < df['player2_avg_time'],
        'player1',
        'player2'
    )
    
    # 编码胜者
    df['player1_won'] = (df['winner'] == 'player1').astype(int)
    df['player2_won'] = (df['winner'] == 'player2').astype(int)
    df['is_draw'] = (df['winner'] == 'draw').astype(int)
    
    # 对局长度分类
    df['game_length_category'] = pd.cut(
        df['total_moves'],
        bins=[0, 20, 40, 60, np.inf],
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    print(f"✓ Added {len(df.columns) - len(pd.read_csv(data_path).columns)} derived features")
    
    # 3. 数据验证
    print(f"\n📊 Data Validation:")
    print(f"   Total records: {len(df)}")
    unique_algos = len(set(df['player1'].unique()) | set(df['player2'].unique()))
    print(f"   Unique algorithms: {unique_algos}")
    print(f"   Winner distribution: {df['winner'].value_counts().to_dict()}")
    print(f"   Avg moves: {df['total_moves'].mean():.1f} ± {df['total_moves'].std():.1f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df


def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    """保存预处理后的数据"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved preprocessed data to {output_path}")


# ==================== 主函数 ====================

def main():
    """主程序"""
    print("=" * 60)
    print(" Performance Analysis Pipeline")
    print("=" * 60)
    
    # 1. 找到最新的数据文件
    data_files = glob("./data/results/self_play/aggregated/results_*.csv")
    
    if not data_files:
        print("\n❌ No data files found!")
        print("   Please run scripts/eval_models.py first to generate data.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    
    # 2. 数据预处理
    print("\n" + "=" * 60)
    print(" STEP 1: Data Preprocessing")
    print("=" * 60)
    
    preprocessed_data = preprocess_data(latest_file)
    
    # 保存预处理后的数据
    output_path = "./data/results/self_play/preprocessed_data.csv"
    save_preprocessed_data(preprocessed_data, output_path)
    
    # 打印汇总统计
    print("\n📊 Summary Statistics:")
    numeric_cols = ['total_moves', 'player1_avg_time', 'player2_avg_time', 'total_time']
    summary = preprocessed_data[numeric_cols].describe()
    print(summary)
    
    # 3. 统计分析
    print("\n" + "=" * 60)
    print(" STEP 2: Statistical Analysis")
    print("=" * 60)
    
    analyzer = StatisticalAnalyzer(preprocessed_data)
    
    # 3.1 胜率统计
    win_rates = analyzer.calculate_win_rates()
    win_rates.to_csv("./data/results/win_rates.csv", index=False)
    print(f"\n✓ Saved win rates to ./data/results/win_rates.csv")
    
    # 3.2 响应时间统计
    time_stats = analyzer.calculate_response_times()
    time_stats.to_csv("./data/results/response_times.csv", index=False)
    print(f"✓ Saved response times to ./data/results/response_times.csv")
    
    # 3.3 对战矩阵
    matchup_matrix = analyzer.generate_matchup_matrix()
    matchup_matrix.to_csv("./data/results/matchup_matrix.csv")
    print(f"✓ Saved matchup matrix to ./data/results/matchup_matrix.csv")
    
    # 3.4 显著性检验
    significance_tests = analyzer.run_all_pairwise_tests()
    significance_tests.to_csv("./data/results/significance_tests.csv", index=False)
    print(f"✓ Saved significance tests to ./data/results/significance_tests.csv")
    
    # 3.5 ELO评分
    elo_ratings = analyzer.calculate_elo_ratings(k_factor=32.0)
    elo_ratings.to_csv("./data/results/elo_ratings.csv", index=False)
    print(f"✓ Saved ELO ratings to ./data/results/elo_ratings.csv")
    
    # 4. 总结
    print("\n" + "=" * 60)
    print(" Analysis Complete!")
    print("=" * 60)
    print("\n✅ All analysis results saved to ./data/results/")
    print("\nGenerated files:")
    print("  - preprocessed_data.csv      : Cleaned and standardized data")
    print("  - win_rates.csv              : Win rate statistics")
    print("  - response_times.csv         : Response time statistics")
    print("  - matchup_matrix.csv         : Head-to-head win rates")
    print("  - significance_tests.csv     : Statistical significance tests")
    print("  - elo_ratings.csv            : ELO ratings")
    
    print("\n🎯 Next step: Run scripts/generate_visualizations.py to create charts")


if __name__ == "__main__":
    main()
