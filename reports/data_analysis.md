# Data Analysis Report (Person E)

**Generated**: 2025-12-11 19:44:15

## 1. Experiment Overview

This report analyzes the self-play evaluation results of multiple Gomoku AI algorithms, including win rate analysis, response time evaluation, and statistical significance testing.

## 2. Algorithm Performance Rankings

### 2.1 Win Rate Rankings

| algorithm    |   win_rate |   wins |   losses |   draws |
|:-------------|-----------:|-------:|---------:|--------:|
| AlphaBeta-D2 |   0.875    |     70 |       10 |       0 |
| Minimax-D2   |   0.875    |     70 |       10 |       0 |
| Greedy       |   0.5      |     40 |       40 |       0 |
| MCTS-100     |   0.128205 |     10 |       68 |       0 |
| DQN          |   0.102564 |      8 |       70 |       0 |

### 2.2 ELO Ratings

| algorithm    |   elo_rating |
|:-------------|-------------:|
| AlphaBeta-D2 |      1866.51 |
| Minimax-D2   |      1772.72 |
| Greedy       |      1476.29 |
| DQN          |      1233.02 |
| MCTS-100     |      1151.46 |

### 2.3 Response Times

| algorithm    |   mean_time |   median_time |   std_time |
|:-------------|------------:|--------------:|-----------:|
| DQN          |  0.00196007 |    0.00170647 | 0.00079402 |
| Greedy       |  0.063444   |    0.0387802  | 0.0739923  |
| AlphaBeta-D2 |  1.10279    |    0.84634    | 0.95227    |
| Minimax-D2   |  1.12253    |    0.823202   | 0.991491   |
| MCTS-100     | 11.233      |   11.2951     | 1.5698     |

## 3. Key Findings

### 3.1 Best Performing Algorithms
- **Highest Win Rate**: AlphaBeta-D2 (87.5%)
- **Fastest Response**: DQN (0.0020s)
- **Highest ELO**: AlphaBeta-D2 (ELO 1867)

### 3.2 Efficiency Analysis
- Top 3 fastest algorithms:
  DQN, Greedy, AlphaBeta-D2

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
2. **Efficiency**: DQN shows clear advantages in response speed
3. **Overall**: According to the ELO rating system, AlphaBeta-D2 has the strongest comprehensive ability

## 6. Recommendations

- For win-rate focused scenarios, recommend using AlphaBeta-D2
- For real-time gameplay requirements, recommend using DQN
- For balanced performance and efficiency, AlphaBeta-D2 is the best choice

---

*This report was automatically generated | Person E*
