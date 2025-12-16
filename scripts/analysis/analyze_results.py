"""Data Analysis Script - Process 5-thread evaluation results and generate statistics

Generates data files for visualization from raw 5-thread evaluation results:
- preprocessed_data.csv
- win_rates.csv
- response_times.csv
- matchup_matrix.csv
- elo_ratings.csv
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple


def find_latest_result_file(results_dir: str) -> Path:
    """Find the latest evaluation result file."""
    results_path = Path(results_dir)
    
    # Find CSV files
    csv_files = list(results_path.glob("results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    # Return latest file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Found latest result file: {latest_file.name}")
    return latest_file


def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate win rate statistics for each AI."""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    win_rates_data = []
    
    for ai in ai_names:
        # Records as P1
        df_as_p1 = df[df['player1'] == ai]
        # Records as P2
        df_as_p2 = df[df['player2'] == ai]
        
        # Calculate Wins
        p1_wins = len(df_as_p1[df_as_p1['winner'] == 'player1'])
        p2_wins = len(df_as_p2[df_as_p2['winner'] == 'player2'])
        total_wins = p1_wins + p2_wins
        
        # Calculate Losses
        p1_losses = len(df_as_p1[df_as_p1['winner'] == 'player2'])
        p2_losses = len(df_as_p2[df_as_p2['winner'] == 'player1'])
        total_losses = p1_losses + p2_losses
        
        # Calculate Draws
        p1_draws = len(df_as_p1[df_as_p1['winner'] == 'draw'])
        p2_draws = len(df_as_p2[df_as_p2['winner'] == 'draw'])
        total_draws = p1_draws + p2_draws
        
        # Total Games
        total_games = len(df_as_p1) + len(df_as_p2)
        
        # Win Rate
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
    """Calculate response time statistics for each AI."""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    
    time_stats_data = []
    
    for ai in ai_names:
        # Collect all response times for this AI
        as_p1_times = df[df['player1'] == ai]['player1_avg_time']
        as_p2_times = df[df['player2'] == ai]['player2_avg_time']
        all_times = pd.concat([as_p1_times, as_p2_times])
        
        time_stats_data.append({
            'algorithm': ai,
            'mean_time': all_times.mean(),
            'median_time': all_times.median(),
            'std_time': all_times.std(),
            'min_time': all_times.min(),
            'max_time': all_times.max(),
            'count': len(all_times)
        })
        
    return pd.DataFrame(time_stats_data)


def create_matchup_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a win-rate matrix (Row vs Column)."""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    matrix = pd.DataFrame(index=ai_names, columns=ai_names, dtype=float)
    
    for ai1 in ai_names:
        for ai2 in ai_names:
            if ai1 == ai2:
                matrix.loc[ai1, ai2] = np.nan
                continue
                
            # Games where AI1 is P1 vs AI2
            games1 = df[(df['player1'] == ai1) & (df['player2'] == ai2)]
            # Games where AI1 is P2 vs AI2
            games2 = df[(df['player2'] == ai1) & (df['player1'] == ai2)]
            
            wins = 0
            wins += len(games1[games1['winner'] == 'player1'])
            wins += len(games2[games2['winner'] == 'player2'])
            
            total = len(games1) + len(games2)
            
            if total > 0:
                matrix.loc[ai1, ai2] = wins / total
            else:
                matrix.loc[ai1, ai2] = np.nan
                
    return matrix


def calculate_elo(df: pd.DataFrame, initial_rating=1500, k_factor=32) -> pd.DataFrame:
    """Calculate ELO ratings."""
    ai_names = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    ratings = {ai: initial_rating for ai in ai_names}
    
    # Sort games by time (rough approximation by index as they are appended chronologically)
    sorted_df = df.reset_index(drop=True)
    
    for _, row in sorted_df.iterrows():
        p1 = row['player1']
        p2 = row['player2']
        winner = row['winner']
        
        r1 = ratings[p1]
        r2 = ratings[p2]
        
        # Expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        if winner == 'player1':
            s1, s2 = 1, 0
        elif winner == 'player2':
            s1, s2 = 0, 1
        else:
            s1, s2 = 0.5, 0.5
            
        # Update ratings
        ratings[p1] = r1 + k_factor * (s1 - e1)
        ratings[p2] = r2 + k_factor * (s2 - e2)
        
    # Convert to DataFrame
    elo_df = pd.DataFrame(list(ratings.items()), columns=['algorithm', 'elo_rating'])
    elo_df = elo_df.sort_values('elo_rating', ascending=False)
    
    return elo_df


def analyze_results(input_dir: str, output_dir: str):
    """Main analysis function."""
    print(f"Analysing results from: {input_dir}")
    
    # 1. Load Data
    data_file = find_latest_result_file(input_dir)
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} game records")
    
    # 2. Create Output Directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Save Raw Data (Preprocessed)
    # Ensure standard names
    df.to_csv(out_path / "preprocessed_data.csv", index=False)
    
    # 4. Calculate Win Rates
    win_rates = calculate_win_rates(df)
    win_rates.to_csv(out_path / "win_rates.csv", index=False)
    print("[OK] Win rates calculated")
    
    # 5. Response Times
    response_times = calculate_response_times(df)
    response_times.to_csv(out_path / "response_times.csv", index=False)
    print("[OK] Response times calculated")
    
    # 6. Matchup Matrix
    matrix = create_matchup_matrix(df)
    matrix.to_csv(out_path / "matchup_matrix.csv")
    print("[OK] Matchup matrix created")

    # 7. ELO Ratings
    elo = calculate_elo(df)
    elo.to_csv(out_path / "elo_ratings.csv", index=False)
    print("[OK] ELO ratings calculated")
    
    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print(win_rates[['algorithm', 'win_rate', 'total_games']].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Gomoku Evaluation Results")
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing results_*.csv')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for analyzed data')
    
    args = parser.parse_args()
    
    analyze_results(args.input_dir, args.output_dir)
