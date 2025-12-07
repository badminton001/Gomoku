# Data Analysis Report (Person E)

**Generated**: 2025-12-08 00:13:57

## 1. Experiment Overview

This report analyzes the self-play evaluation results of multiple Gomoku AI algorithms, including win rate analysis, response time evaluation, and statistical significance testing.

## 2. Algorithm Performance Rankings

### 2.1 Win Rate Rankings

| algorithm    |   win_rate |   wins |   losses |   draws |
|:-------------|-----------:|-------:|---------:|--------:|
| AlphaBeta-D2 |       0.75 |     60 |       20 |       0 |
| Minimax-D2   |       0.75 |     60 |       20 |       0 |
| Greedy       |       0    |      0 |       80 |       0 |

### 2.2 ELO Ratings

| algorithm    |   elo_rating |
|:-------------|-------------:|
| AlphaBeta-D2 |      1803.95 |
| Minimax-D2   |      1575.6  |
| Greedy       |      1120.45 |

### 2.3 Response Times

| algorithm    |   mean_time |   median_time |   std_time |
|:-------------|------------:|--------------:|-----------:|
| Greedy       |    0.086273 |     0.0488339 |  0.0892807 |
| Minimax-D2   |    2.44903  |     1.5049    |  2.04698   |
| AlphaBeta-D2 |    2.83457  |     1.34678   |  2.91674   |

## 3. Key Findings

### 3.1 Best Performing Algorithms
- **Highest Win Rate**: AlphaBeta-D2 (75.0%)
- **Fastest Response**: Greedy (0.0863s)
- **Highest ELO**: AlphaBeta-D2 (ELO 1804)

### 3.2 Efficiency Analysis
- Top 3 fastest algorithms:
  Greedy, Minimax-D2, AlphaBeta-D2

## 4. Visualizations

This analysis generated 15 high-quality visualization charts, located in the `./data/results/visualizations/` directory.

Main charts include:
- Matchup matrix heatmap
- Win rate comparison bar chart
- Response time box plot
- ELO rating chart
- Performance radar chart

## 5. Conclusions

Based on the evaluation results, we can conclude:

1. **Performance**: AlphaBeta-D2 demonstrates the best win rate
2. **Efficiency**: Greedy shows clear advantages in response speed
3. **Overall**: According to the ELO rating system, AlphaBeta-D2 has the strongest comprehensive ability

## 6. Recommendations

- For win-rate focused scenarios, recommend using AlphaBeta-D2
- For real-time gameplay requirements, recommend using Greedy
- For balanced performance and efficiency, AlphaBeta-D2 is the best choice

---

*This report was automatically generated | Person E*
