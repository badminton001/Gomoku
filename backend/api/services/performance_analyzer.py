"""Statistical Analysis Module

Responsible for win rate statistics, response time analysis, and significance testing.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import itertools


class StatisticalAnalyzer:
    """Statistical Analyzer"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize Statistical Analyzer
        
        Args:
            data: Preprocessed data
        """
        self.data = data
        print(f"[OK] Initialized analyzer with {len(data)} records")
    
    def calculate_win_rates(self) -> pd.DataFrame:
        """Calculate win rates for each algorithm
        
        Returns:
            Win rate statistics DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        win_stats = []
        for ai in ai_names:
            # Records as player1
            as_p1 = self.data[self.data['player1'] == ai]
            p1_wins = (as_p1['winner'] == 'player1').sum()
            p1_draws = (as_p1['winner'] == 'draw').sum()
            
            # Records as player2
            as_p2 = self.data[self.data['player2'] == ai]
            p2_wins = (as_p2['winner'] == 'player2').sum()
            p2_draws = (as_p2['winner'] == 'draw').sum()
            
            total_games = len(as_p1) + len(as_p2)
            total_wins = p1_wins + p2_wins
            total_draws = p1_draws + p2_draws
            total_losses = total_games - total_wins - total_draws
            
            win_stats.append({
                'algorithm': ai,
                'total_games': total_games,
                'wins': total_wins,
                'losses': total_losses,
                'draws': total_draws,
                'win_rate': total_wins / total_games if total_games > 0 else 0,
                'draw_rate': total_draws / total_games if total_games > 0 else 0,
                'win_loss_ratio': total_wins / total_losses if total_losses > 0 else float('inf')
            })
        
        df = pd.DataFrame(win_stats).sort_values('win_rate', ascending=False)
        
        print(f"\n[INFO] Win Rate Rankings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | Win Rate: {row['win_rate']:.1%} "
                  f"({row['wins']}W-{row['losses']}L-{row['draws']}D)")
        
        return df
    
    def calculate_response_times(self) -> pd.DataFrame:
        """Calculate response time statistics
        
        Returns:
            Response time statistics DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        time_stats = []
        for ai in ai_names:
            # Collect all response times for this AI
            as_p1 = self.data[self.data['player1'] == ai]['player1_avg_time']
            as_p2 = self.data[self.data['player2'] == ai]['player2_avg_time']
            all_times = pd.concat([as_p1, as_p2])
            
            time_stats.append({
                'algorithm': ai,
                'mean_time': all_times.mean(),
                'median_time': all_times.median(),
                'std_time': all_times.std(),
                'min_time': all_times.min(),
                'max_time': all_times.max(),
                'q25_time': all_times.quantile(0.25),
                'q75_time': all_times.quantile(0.75)
            })
        
        df = pd.DataFrame(time_stats).sort_values('mean_time')
        
        print(f"\n[INFO] Response Time Rankings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | Mean: {row['mean_time']:.4f}s "
                  f"(Median: {row['median_time']:.4f}s)")
        
        return df
    
    def pairwise_significance_test(self, ai1: str, ai2: str) -> Tuple[float, float, str]:
        """Pairwise algorithm significance test (Mann-Whitney U test)
        
        Args:
            ai1: Name of algorithm 1
            ai2: Name of algorithm 2
            
        Returns:
            (Statistic, p-value, Conclusion)
        """
        # ai1 performance data (1=win, 0=loss)
        ai1_as_p1 = self.data[(self.data['player1'] == ai1) & (self.data['player2'] == ai2)]
        ai1_wins_p1 = (ai1_as_p1['winner'] == 'player1').astype(int)
        
        ai1_as_p2 = self.data[(self.data['player1'] == ai2) & (self.data['player2'] == ai1)]
        ai1_wins_p2 = (ai1_as_p2['winner'] == 'player2').astype(int)
        
        ai1_performance = pd.concat([ai1_wins_p1, ai1_wins_p2])
        
        # ai2 performance data
        ai2_as_p1 = self.data[(self.data['player1'] == ai2) & (self.data['player2'] == ai1)]
        ai2_wins_p1 = (ai2_as_p1['winner'] == 'player1').astype(int)
        
        ai2_as_p2 = self.data[(self.data['player1'] == ai1) & (self.data['player2'] == ai2)]
        ai2_wins_p2 = (ai2_as_p2['winner'] == 'player2').astype(int)
        
        ai2_performance = pd.concat([ai2_wins_p1, ai2_wins_p2])
        
        if len(ai1_performance) == 0 or len(ai2_performance) == 0:
            return 0.0, 1.0, "Insufficient data"
        
        # specific Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            ai1_performance, 
            ai2_performance, 
            alternative='two-sided'
        )
        
        # Determine significance
        alpha = 0.05
        if p_value < alpha:
            conclusion = f"Significant (p={p_value:.4f} < {alpha})"
        else:
            conclusion = f"Not significant (p={p_value:.4f} >= {alpha})"
        
        return statistic, p_value, conclusion
    
    def generate_matchup_matrix(self) -> pd.DataFrame:
        """Generate matchup matrix (row vs column win rate)
        
        Returns:
            Matchup matrix DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        matrix = pd.DataFrame(0.0, index=ai_names, columns=ai_names)
        
        for ai1 in ai_names:
            for ai2 in ai_names:
                if ai1 == ai2:
                    matrix.loc[ai1, ai2] = np.nan
                    continue
                
                # ai1 vs ai2 matches
                matches = self.data[
                    ((self.data['player1'] == ai1) & (self.data['player2'] == ai2)) |
                    ((self.data['player1'] == ai2) & (self.data['player2'] == ai1))
                ]
                
                if len(matches) == 0:
                    matrix.loc[ai1, ai2] = np.nan
                    continue
                
                ai1_wins = (
                    ((matches['player1'] == ai1) & (matches['winner'] == 'player1')) |
                    ((matches['player2'] == ai1) & (matches['winner'] == 'player2'))
                ).sum()
                
                total = len(matches)
                matrix.loc[ai1, ai2] = ai1_wins / total
        
        print(f"\n[INFO] Matchup Matrix (row vs column win rate):")
        print(matrix.round(3))
        
        return matrix
    
    def run_all_pairwise_tests(self) -> pd.DataFrame:
        """Run significance tests for all algorithm pairs
        
        Returns:
            Significance test results DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        results = []
        for ai1, ai2 in itertools.combinations(ai_names, 2):
            stat, p_val, conclusion = self.pairwise_significance_test(ai1, ai2)
            results.append({
                'algorithm_1': ai1,
                'algorithm_2': ai2,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'conclusion': conclusion
            })
        
        df = pd.DataFrame(results).sort_values('p_value')
        
        print(f"\n[INFO] Pairwise Significance Tests:")
        for idx, row in df.iterrows():
            sig_marker = "[*]" if row['significant'] else "[ ]"
            print(f"   {sig_marker} {row['algorithm_1']} vs {row['algorithm_2']}: {row['conclusion']}")
        
        return df
    
    def calculate_elo_ratings(self, k_factor: float = 32.0) -> pd.DataFrame:
        """Calculate ELO ratings (Chess rating system)
        
        Args:
            k_factor: K factor, controls magnitude of rating changes
            
        Returns:
            ELO ratings DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        # Initialize ELO ratings (all start at 1500)
        elo = {ai: 1500.0 for ai in ai_names}
        
        # Process games chronologically
        for _, game in self.data.iterrows():
            p1 = game['player1']
            p2 = game['player2']
            
            # Expected scores
            expected_p1 = 1 / (1 + 10 ** ((elo[p2] - elo[p1]) / 400))
            expected_p2 = 1 - expected_p1
            
            # Actual scores
            if game['winner'] == 'player1':
                actual_p1, actual_p2 = 1.0, 0.0
            elif game['winner'] == 'player2':
                actual_p1, actual_p2 = 0.0, 1.0
            else:  # draw
                actual_p1, actual_p2 = 0.5, 0.5
            
            # Update ELO
            elo[p1] += k_factor * (actual_p1 - expected_p1)
            elo[p2] += k_factor * (actual_p2 - expected_p2)
        
        df = pd.DataFrame([
            {'algorithm': ai, 'elo_rating': rating}
            for ai, rating in elo.items()
        ]).sort_values('elo_rating', ascending=False)
        
        print(f"\n[INFO] ELO Ratings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | ELO: {row['elo_rating']:.0f}")
        
        return df
