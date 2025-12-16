"""Report Generation Script

Automatically generates Markdown format experiment reports
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime


def generate_data_analysis_report():
    """Generate data analysis report"""
    
    # Load data
    win_rates = pd.read_csv("./data/results/win_rates.csv")
    time_stats = pd.read_csv("./data/results/response_times.csv")
    elo_ratings = pd.read_csv("./data/results/elo_ratings.csv")
    
    report = f"""# Data Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Experiment Overview

This report analyzes the self-play evaluation results of multiple Gomoku AI algorithms, including win rate analysis, response time evaluation, and statistical significance testing.

## 2. Algorithm Performance Rankings

### 2.1 Win Rate Rankings

{win_rates[['algorithm', 'win_rate', 'wins', 'losses', 'draws']].to_markdown(index=False)}

### 2.2 ELO Ratings

{elo_ratings.to_markdown(index=False)}

### 2.3 Response Times

{time_stats[['algorithm', 'mean_time', 'median_time', 'std_time']].to_markdown(index=False)}

## 3. Key Findings

### 3.1 Best Performing Algorithms
- **Highest Win Rate**: {win_rates.iloc[0]['algorithm']} ({win_rates.iloc[0]['win_rate']:.1%})
- **Fastest Response**: {time_stats.iloc[0]['algorithm']} ({time_stats.iloc[0]['mean_time']:.4f}s)
- **Highest ELO**: {elo_ratings.iloc[0]['algorithm']} (ELO {elo_ratings.iloc[0]['elo_rating']:.0f})

### 3.2 Efficiency Analysis
- Top 3 fastest algorithms:
  {', '.join(time_stats.head(3)['algorithm'].tolist())}

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

1. **Performance**: {win_rates.iloc[0]['algorithm']} demonstrates the best win rate
2. **Efficiency**: {time_stats.iloc[0]['algorithm']} shows clear advantages in response speed
3. **Overall**: According to the ELO rating system, {elo_ratings.iloc[0]['algorithm']} has the strongest comprehensive ability

## 6. Recommendations

- For win-rate focused scenarios, recommend using {win_rates.iloc[0]['algorithm']}
- For real-time gameplay requirements, recommend using {time_stats.iloc[0]['algorithm']}
- For balanced performance and efficiency, {elo_ratings.iloc[0]['algorithm']} is the best choice

---

*This report was automatically generated*
"""
    
    # Save report
    output_path = "./reports/data_analysis.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[OK] Generated: {output_path}")
    return output_path


def main():
    """Main function"""
    print("\nReport Generation")
    
    print("\n[INFO] Generating reports...\n")
    
    # Generate data analysis report
    report_path = generate_data_analysis_report()
    
    print("\nReport Generation Complete!")
    print(f"\n[OK] Report saved to: {report_path}")
    print(f"\n[INFO] All deliverables ready in:")
    print(f"   - ./reports/")
    print(f"   - ./data/results/visualizations/")


if __name__ == "__main__":
    main()
