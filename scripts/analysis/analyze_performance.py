"""
Performance Analysis Pipeline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

from backend.services.performance_analyzer import StatisticalAnalyzer


# Preprocessing

def preprocess_data(data_path: str) -> pd.DataFrame:
    """Preprocess data."""
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    original_count = len(df)
    
    # 1. Cleaning
    # Remove outliers
    q99 = df['total_moves'].quantile(0.99)
    df = df[df['total_moves'] <= q99]
    
    # Remove timeouts
    df = df[df['player1_avg_time'] < 60]
    df = df[df['player2_avg_time'] < 60]
    
    # Drop missing
    df = df.dropna()
    
    # Validate winner
    valid_winners = ['player1', 'player2', 'draw']
    df = df[df['winner'].isin(valid_winners)]
    
    cleaned_count = len(df)
    removed = original_count - cleaned_count
    print(f"Cleaned: {original_count} -> {cleaned_count} records (removed {removed})")
    
    # 2. Features
    df['total_time'] = df['player1_avg_time'] + df['player2_avg_time']
    df['time_difference'] = np.abs(df['player1_avg_time'] - df['player2_avg_time'])
    df['faster_player'] = np.where(
        df['player1_avg_time'] < df['player2_avg_time'],
        'player1',
        'player2'
    )
    
    # Encode values
    df['player1_won'] = (df['winner'] == 'player1').astype(int)
    df['player2_won'] = (df['winner'] == 'player2').astype(int)
    df['is_draw'] = (df['winner'] == 'draw').astype(int)
    
    # Categorize length
    df['game_length_category'] = pd.cut(
        df['total_moves'],
        bins=[0, 20, 40, 60, np.inf],
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    print(f"Added {len(df.columns) - len(pd.read_csv(data_path).columns)} derived features")
    
    # 3. Validation
    print(f"\nData Validation:")
    print(f"   Total records: {len(df)}")
    unique_algos = len(set(df['player1'].unique()) | set(df['player2'].unique()))
    print(f"   Unique algorithms: {unique_algos}")
    print(f"   Winner distribution: {df['winner'].value_counts().to_dict()}")
    print(f"   Avg moves: {df['total_moves'].mean():.1f} ± {df['total_moves'].std():.1f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df


def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    """Save CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved preprocessed data to {output_path}")


# Main Execution

def main():
    print("\nPerformance Analysis Pipeline")
    
    # 1. Find latest
    data_files = glob("./data/results/self_play/aggregated/results_*.csv")
    
    if not data_files:
        print("\nNo data files found!")
        print("   Please run scripts/eval_models.py first to generate data.")
        return
        
    latest_file = max(data_files, key=os.path.getctime)
    
    # 2. Preprocess
    df = preprocess_data(latest_file)
    save_preprocessed_data(df, "./data/results/self_play/preprocessed_data.csv")
    
    # 3. Analysis
    analyzer = StatisticalAnalyzer(df)
    
    # Win Rates
    win_rates, matchup_matrix = analyzer.analyze_win_rates()
    
    # Time Stats
    time_stats = analyzer.analyze_response_times()
    
    # ELO
    elo_ratings = analyzer.calculate_elo()
    
    # 4. Save
    win_rates.to_csv("./data/results/win_rates.csv", index=False)
    time_stats.to_csv("./data/results/response_times.csv", index=False)
    matchup_matrix.to_csv("./data/results/matchup_matrix.csv")
    elo_ratings.to_csv("./data/results/elo_ratings.csv", index=False)
    
    print("\nAnalysis Complete!")
    print(" Metrics saved to ./data/results/")


if __name__ == "__main__":
    main()
