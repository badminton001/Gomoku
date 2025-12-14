"""统计分析模块

负责胜率统计、响应时间分析和显著性检验
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import itertools


class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """初始化统计分析器
        
        Args:
            data: 预处理后的数据
        """
        self.data = data
        print(f"✓ Initialized analyzer with {len(data)} records")
    
    def calculate_win_rates(self) -> pd.DataFrame:
        """计算每个算法的胜率
        
        Returns:
            胜率统计DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        win_stats = []
        for ai in ai_names:
            # 作为player1的记录
            as_p1 = self.data[self.data['player1'] == ai]
            p1_wins = (as_p1['winner'] == 'player1').sum()
            p1_draws = (as_p1['winner'] == 'draw').sum()
            
            # 作为player2的记录
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
        
        print(f"\n🏆 Win Rate Rankings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | Win Rate: {row['win_rate']:.1%} "
                  f"({row['wins']}W-{row['losses']}L-{row['draws']}D)")
        
        return df
    
    def calculate_response_times(self) -> pd.DataFrame:
        """计算响应时间统计
        
        Returns:
            响应时间统计DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        time_stats = []
        for ai in ai_names:
            # 收集该AI的所有响应时间
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
        
        print(f"\n⚡ Response Time Rankings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | Mean: {row['mean_time']:.4f}s "
                  f"(Median: {row['median_time']:.4f}s)")
        
        return df
    
    def pairwise_significance_test(self, ai1: str, ai2: str) -> Tuple[float, float, str]:
        """两两算法显著性检验（Mann-Whitney U test）
        
        Args:
            ai1: 算法1名称
            ai2: 算法2名称
            
        Returns:
            (统计量, p值, 结论)
        """
        # ai1的表现数据（1=胜，0=负）
        ai1_as_p1 = self.data[(self.data['player1'] == ai1) & (self.data['player2'] == ai2)]
        ai1_wins_p1 = (ai1_as_p1['winner'] == 'player1').astype(int)
        
        ai1_as_p2 = self.data[(self.data['player1'] == ai2) & (self.data['player2'] == ai1)]
        ai1_wins_p2 = (ai1_as_p2['winner'] == 'player2').astype(int)
        
        ai1_performance = pd.concat([ai1_wins_p1, ai1_wins_p2])
        
        # ai2的表现数据
        ai2_as_p1 = self.data[(self.data['player1'] == ai2) & (self.data['player2'] == ai1)]
        ai2_wins_p1 = (ai2_as_p1['winner'] == 'player1').astype(int)
        
        ai2_as_p2 = self.data[(self.data['player1'] == ai1) & (self.data['player2'] == ai2)]
        ai2_wins_p2 = (ai2_as_p2['winner'] == 'player2').astype(int)
        
        ai2_performance = pd.concat([ai2_wins_p1, ai2_wins_p2])
        
        if len(ai1_performance) == 0 or len(ai2_performance) == 0:
            return 0.0, 1.0, "Insufficient data"
        
        # 执行Mann-Whitney U检验
        statistic, p_value = stats.mannwhitneyu(
            ai1_performance, 
            ai2_performance, 
            alternative='two-sided'
        )
        
        # 判断显著性
        alpha = 0.05
        if p_value < alpha:
            conclusion = f"Significant (p={p_value:.4f} < {alpha})"
        else:
            conclusion = f"Not significant (p={p_value:.4f} >= {alpha})"
        
        return statistic, p_value, conclusion
    
    def generate_matchup_matrix(self) -> pd.DataFrame:
        """生成对战矩阵（行对列的胜率）
        
        Returns:
            对战矩阵DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        matrix = pd.DataFrame(0.0, index=ai_names, columns=ai_names)
        
        for ai1 in ai_names:
            for ai2 in ai_names:
                if ai1 == ai2:
                    matrix.loc[ai1, ai2] = np.nan
                    continue
                
                # ai1 vs ai2的胜率
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
        
        print(f"\n🎯 Matchup Matrix (row vs column win rate):")
        print(matrix.round(3))
        
        return matrix
    
    def run_all_pairwise_tests(self) -> pd.DataFrame:
        """运行所有算法对的显著性检验
        
        Returns:
            显著性检验结果DataFrame
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
        
        print(f"\n📈 Pairwise Significance Tests:")
        for idx, row in df.iterrows():
            sig_marker = "✓" if row['significant'] else "✗"
            print(f"   {sig_marker} {row['algorithm_1']} vs {row['algorithm_2']}: {row['conclusion']}")
        
        return df
    
    def calculate_elo_ratings(self, k_factor: float = 32.0) -> pd.DataFrame:
        """计算ELO评分（Chess rating system）
        
        Args:
            k_factor: K值，控制评分变化幅度
            
        Returns:
            ELO评分DataFrame
        """
        ai_names = sorted(set(self.data['player1'].unique()) | set(self.data['player2'].unique()))
        
        # 初始化ELO评分（所有从1500开始）
        elo = {ai: 1500.0 for ai in ai_names}
        
        # 按时间顺序处理对局
        for _, game in self.data.iterrows():
            p1 = game['player1']
            p2 = game['player2']
            
            # 期望得分
            expected_p1 = 1 / (1 + 10 ** ((elo[p2] - elo[p1]) / 400))
            expected_p2 = 1 - expected_p1
            
            # 实际得分
            if game['winner'] == 'player1':
                actual_p1, actual_p2 = 1.0, 0.0
            elif game['winner'] == 'player2':
                actual_p1, actual_p2 = 0.0, 1.0
            else:  # draw
                actual_p1, actual_p2 = 0.5, 0.5
            
            # 更新ELO
            elo[p1] += k_factor * (actual_p1 - expected_p1)
            elo[p2] += k_factor * (actual_p2 - expected_p2)
        
        df = pd.DataFrame([
            {'algorithm': ai, 'elo_rating': rating}
            for ai, rating in elo.items()
        ]).sort_values('elo_rating', ascending=False)
        
        print(f"\n🎖️  ELO Ratings:")
        for idx, row in df.iterrows():
            print(f"   {row['algorithm']:20s} | ELO: {row['elo_rating']:.0f}")
        
        return df
