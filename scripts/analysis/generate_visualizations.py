"""Visualization Generation Script

Generates 15+ high-quality visualization charts
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("[ERROR] Plotly is not installed. Please run: pip install plotly")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("[ERROR] Matplotlib/Seaborn not installed. Please run: pip install matplotlib seaborn")
    sys.exit(1)

# Set style for matplotlib
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def create_output_dir():
    """Create output directory"""
    output_dir = "./data/results/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    """Load all analysis results"""
    data = pd.read_csv("./data/results/self_play/preprocessed_data.csv")
    win_rates = pd.read_csv("./data/results/win_rates.csv")
    time_stats = pd.read_csv("./data/results/response_times.csv")
    matchup_matrix = pd.read_csv("./data/results/matchup_matrix.csv", index_col=0)
    elo_ratings = pd.read_csv("./data/results/elo_ratings.csv")
    
    return data, win_rates, time_stats, matchup_matrix, elo_ratings


# Plotly Interactive Charts

def plot_matchup_heatmap(matchup_matrix, output_dir):
    """1. Matchup Matrix Heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=matchup_matrix.values,
        x=matchup_matrix.columns,
        y=matchup_matrix.index,
        colorscale='RdYlGn',
        zmid=0.5,
        text=matchup_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Win Rate")
    ))
    
    fig.update_layout(
        title='Algorithm Matchup Matrix (Row vs Column Win Rate)',
        xaxis_title='Opponent',
        yaxis_title='Algorithm',
        width=800,
        height=700
    )
    
    fig.write_html(f"{output_dir}/01_matchup_heatmap.html")
    print("[OK] [1/15] Matchup heatmap")


def plot_win_rate_comparison(win_rates, output_dir):
    """2. Win Rate Comparison Bar Chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=win_rates['algorithm'],
        y=win_rates['win_rate'],
        text=[f"{x:.1%}" for x in win_rates['win_rate']],
        textposition='auto',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Algorithm Win Rate Comparison',
        xaxis_title='Algorithm',
        yaxis_title='Win Rate',
        yaxis=dict(tickformat='.0%'),
        width=900,
        height=500
    )
    
    fig.write_html(f"{output_dir}/02_win_rate_comparison.html")
    print("[OK] [2/15] Win rate comparison")


def plot_response_time_boxplot(data, time_stats, output_dir):
    """3. Response Time Box Plot"""
    ai_names = time_stats['algorithm'].tolist()
    
    fig = go.Figure()
    
    for ai in ai_names:
        as_p1 = data[data['player1'] == ai]['player1_avg_time']
        as_p2 = data[data['player2'] == ai]['player2_avg_time']
        all_times = pd.concat([as_p1, as_p2])
        
        fig.add_trace(go.Box(
            y=all_times,
            name=ai,
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='Algorithm Response Time Distribution',
        yaxis_title='Response Time (seconds)',
        width=900,
        height=500
    )
    
    fig.write_html(f"{output_dir}/03_response_time_boxplot.html")
    print("[OK] [3/15] Response time boxplot")


def plot_elo_ratings(elo_ratings, output_dir):
    """4. ELO Rating Chart"""
    fig = go.Figure()
    
    colors = px.colors.sequential.Viridis
    
    fig.add_trace(go.Bar(
        x=elo_ratings['algorithm'],
        y=elo_ratings['elo_rating'],
        text=[f"{x:.0f}" for x in elo_ratings['elo_rating']],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='ELO Rating Rankings',
        xaxis_title='Algorithm',
        yaxis_title='ELO Rating',
        width=900,
        height=500
    )
    
    fig.write_html(f"{output_dir}/04_elo_ratings.html")
    print("[OK] [4/15] ELO ratings")


def plot_move_distribution(data, output_dir):
    """5. Move Count Distribution Histogram"""
    fig = px.histogram(
        data,
        x='total_moves',
        nbins=30,
        title='Game Move Count Distribution',
        labels={'total_moves': 'Move Count', 'count': 'Frequency'},
        color_discrete_sequence=['steelblue']
    )
    
    fig.add_vline(
        x=data['total_moves'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {data['total_moves'].mean():.1f}"
    )
    
    fig.update_layout(width=900, height=500)
    fig.write_html(f"{output_dir}/05_move_distribution.html")
    print("[OK] [5/15] Move distribution")


# Matplotlib/Seaborn Static Charts

def plot_algorithm_ranking(win_rates, output_dir):
    """6. Algorithm Ranking Chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("RdYlGn", len(win_rates))
    ax.barh(win_rates['algorithm'], win_rates['win_rate'], color=colors)
    
    for i, (idx, row) in enumerate(win_rates.iterrows()):
        ax.text(row['win_rate'] + 0.01, i, f"{row['win_rate']:.1%}", 
               va='center', fontsize=12)
    
    ax.set_xlabel('Win Rate', fontsize=14)
    ax.set_title('Algorithm Win Rate Ranking', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_algorithm_ranking.png", dpi=300, bbox_inches='tight')
    print("[OK] [6/15] Algorithm ranking")
    plt.close()


def plot_time_vs_winrate(win_rates, time_stats, output_dir):
    """7. Time vs Win Rate Scatter Plot"""
    merged = win_rates.merge(time_stats, on='algorithm')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        merged['mean_time'],
        merged['win_rate'],
        s=200,
        c=merged['win_rate'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    for _, row in merged.iterrows():
        ax.annotate(
            row['algorithm'],
            (row['mean_time'], row['win_rate']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    ax.set_xlabel('Average Response Time (seconds)', fontsize=14)
    ax.set_ylabel('Win Rate', fontsize=14)
    ax.set_title('Algorithm Efficiency vs Performance', fontsize=16, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.colorbar(scatter, label='Win Rate')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_time_vs_winrate.png", dpi=300, bbox_inches='tight')
    print("[OK] [7/15] Time vs winrate")
    plt.close()


def plot_correlation_matrix(data, output_dir):
    """8. Correlation Matrix"""
    numeric_cols = ['total_moves', 'player1_avg_time', 'player2_avg_time', 'total_time']
    corr = data[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_title('Feature Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_correlation_matrix.png", dpi=300, bbox_inches='tight')
    print("[OK] [8/15] Correlation matrix")
    plt.close()


def plot_win_loss_stats(win_rates, output_dir):
    """9. Win-Loss Statistics Stacked Bar Chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(win_rates))
    width = 0.6
    
    ax.bar(x, win_rates['wins'], width, label='Wins', color='green', alpha=0.8)
    ax.bar(x, win_rates['losses'], width, bottom=win_rates['wins'], label='Losses', color='red', alpha=0.8)
    ax.bar(x, win_rates['draws'], width, bottom=win_rates['wins'] + win_rates['losses'], label='Draws', color='gray', alpha=0.8)
    
    ax.set_ylabel('Number of Games', fontsize=14)
    ax.set_title('Algorithm Win-Loss Statistics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(win_rates['algorithm'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_win_loss_stats.png", dpi=300, bbox_inches='tight')
    print("[OK] [9/15] Win-loss stats")
    plt.close()


def plot_time_distribution(time_stats, output_dir):
    """10. Time Distribution Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(time_stats))
    width = 0.35
    
    ax.bar(x - width/2, time_stats['mean_time'], width, label='Mean Time', alpha=0.8)
    ax.bar(x + width/2, time_stats['median_time'], width, label='Median Time', alpha=0.8)
    
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Response Time Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(time_stats['algorithm'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_time_distribution.png", dpi=300, bbox_inches='tight')
    print("[OK] [10/15] Time distribution")
    plt.close()


def plot_game_length_category(data, output_dir):
    """11. Game Length Category"""
    category_counts = data['game_length_category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Game Length Categories', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_game_length_category.png", dpi=300, bbox_inches='tight')
    print("[OK] [11/15] Game length category")
    plt.close()


def plot_algorithm_winrate_trend(data, output_dir):
    """12. Win Rate Trend (Cumulative)"""
    algorithms = sorted(set(data['player1'].unique()) | set(data['player2'].unique()))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algo in algorithms:
        cumulative_wins = []
        cumulative_total = []
        
        for idx, row in data.iterrows():
            if row['player1'] == algo or row['player2'] == algo:
                won = (row['player1'] == algo and row['winner'] == 'player1') or \
                      (row['player2'] == algo and row['winner'] == 'player2')
                cumulative_wins.append(1 if won else 0)
                cumulative_total.append(1)
        
        if cumulative_wins:
            cumsum_wins = np.cumsum(cumulative_wins)
            cumsum_total = np.cumsum(cumulative_total)
            winrate_trend = cumsum_wins / cumsum_total
            ax.plot(range(len(winrate_trend)), winrate_trend, label=algo, linewidth=2)
    
    ax.set_xlabel('Game Number', fontsize=14)
    ax.set_ylabel('Cumulative Win Rate', fontsize=14)
    ax.set_title('Algorithm Cumulative Win Rate Trend', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/12_winrate_trend.png", dpi=300, bbox_inches='tight')
    print("[OK] [12/15] Winrate trend")
    plt.close()


def plot_performance_radar(win_rates, time_stats, output_dir):
    """13. Performance Radar Chart"""
    merged = win_rates.merge(time_stats, on='algorithm')
    
    # Normalize metrics (0-1)
    merged['win_rate_norm'] = merged['win_rate']
    merged['speed_norm'] = 1 - (merged['mean_time'] - merged['mean_time'].min()) / (merged['mean_time'].max() - merged['mean_time'].min() + 0.001)
    merged['stability_norm'] = 1 - (merged['std_time'] - merged['std_time'].min()) / (merged['std_time'].max() - merged['std_time'].min() + 0.001)
    
    categories = ['Win Rate', 'Speed', 'Stability']
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for _, row in merged.iterrows():
        values = [row['win_rate_norm'], row['speed_norm'], row['stability_norm']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['algorithm'])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Algorithm Performance Radar Chart', size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/13_performance_radar.png", dpi=300, bbox_inches='tight')
    print("[OK] [13/15] Performance radar")
    plt.close()


def plot_moves_vs_time(data, output_dir):
    """14. Moves vs Time Scatter Plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(data['total_moves'], data['total_time'], alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(data['total_moves'], data['total_time'], 1)
    p = np.poly1d(z)
    ax.plot(data['total_moves'], p(data['total_moves']), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Total Moves', fontsize=14)
    ax.set_ylabel('Total Time (seconds)', fontsize=14)
    ax.set_title('Game Moves vs Time Relationship', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/14_moves_vs_time.png", dpi=300, bbox_inches='tight')
    print("[OK] [14/15] Moves vs time")
    plt.close()


def plot_summary_dashboard(win_rates, time_stats, elo_ratings, output_dir):
    """15. Summary Dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Win rate bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.barh(win_rates['algorithm'], win_rates['win_rate'], color='skyblue')
    ax1.set_xlabel('Win Rate')
    ax1.set_title('Algorithm Win Rate Ranking', fontweight='bold')
    ax1.set_xlim(0, 1)
    
    # ELO ratings
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(range(len(elo_ratings)), elo_ratings['elo_rating'], color='lightgreen')
    ax2.set_xticks(range(len(elo_ratings)))
    ax2.set_xticklabels(elo_ratings['algorithm'], rotation=45, ha='right')
    ax2.set_ylabel('ELO Rating')
    ax2.set_title('ELO Ratings', fontweight='bold')
    
    # Response time
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(time_stats))
    ax3.bar(x, time_stats['mean_time'], alpha=0.7, color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(time_stats['algorithm'])
    ax3.set_ylabel('Average Response Time (seconds)')
    ax3.set_title('Response Time Comparison', fontweight='bold')
    
    # Win-loss-draw statistics
    ax4 = fig.add_subplot(gs[2, :])
    x = np.arange(len(win_rates))
    width = 0.6
    ax4.bar(x, win_rates['wins'], width, label='Wins', color='green', alpha=0.7)
    ax4.bar(x, win_rates['losses'], width, bottom=win_rates['wins'], label='Losses', color='red', alpha=0.7)
    ax4.bar(x, win_rates['draws'], width, bottom=win_rates['wins'] + win_rates['losses'], label='Draws', color='gray', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(win_rates['algorithm'])
    ax4.set_ylabel('Number of Games')
    ax4.set_title('Win-Loss-Draw Statistics', fontweight='bold')
    ax4.legend()
    
    plt.suptitle('Gomoku AI Algorithm Evaluation Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(f"{output_dir}/15_summary_dashboard.png", dpi=300, bbox_inches='tight')
    print("[OK] [15/15] Summary dashboard")
    plt.close()


# Main Function

def main():
    """Generate all visualizations"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate Visualizations")
    parser.add_argument('--input-dir', type=str, default="./data/results", help="Input directory containing CSVs")
    parser.add_argument('--output-dir', type=str, default="./data/results/visualizations", help="Output directory for charts")
    args = parser.parse_args()

    print("\nVisualization Generation")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}\n")
    
    # Load data
    print(f"[INFO] Loading data from {args.input_dir}...")
    try:
        data = pd.read_csv(os.path.join(args.input_dir, "preprocessed_data.csv"))
        win_rates = pd.read_csv(os.path.join(args.input_dir, "win_rates.csv"))
        time_stats = pd.read_csv(os.path.join(args.input_dir, "response_times.csv"))
        matchup_matrix = pd.read_csv(os.path.join(args.input_dir, "matchup_matrix.csv"), index_col=0)
        elo_ratings = pd.read_csv(os.path.join(args.input_dir, "elo_ratings.csv"))
        print("[OK] Data loaded\n")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return

    # Generate charts
    print("[INFO] Generating visualizations...\n")
    
    # Plotly interactive charts (5)
    try: plot_matchup_heatmap(matchup_matrix, output_dir)
    except: pass
    try: plot_win_rate_comparison(win_rates, output_dir)
    except: pass
    try: plot_response_time_boxplot(data, time_stats, output_dir)
    except: pass
    try: plot_elo_ratings(elo_ratings, output_dir)
    except: pass
    try: plot_move_distribution(data, output_dir)
    except: pass
    
    # Matplotlib static charts (10)
    try: plot_algorithm_ranking(win_rates, output_dir)
    except: pass
    try: plot_time_vs_winrate(win_rates, time_stats, output_dir)
    except: pass
    try: plot_correlation_matrix(data, output_dir)
    except: pass
    try: plot_win_loss_stats(win_rates, output_dir)
    except: pass
    try: plot_time_distribution(time_stats, output_dir)
    except: pass
    try: plot_game_length_category(data, output_dir)
    except: pass
    try: plot_algorithm_winrate_trend(data, output_dir)
    except: pass
    try: plot_performance_radar(win_rates, time_stats, output_dir)
    except: pass
    try: plot_moves_vs_time(data, output_dir)
    except: pass
    try: plot_summary_dashboard(win_rates, time_stats, elo_ratings, output_dir)
    except: pass
    
    print("\nVisualization Complete!")
    print(f"\n[OK] Generated visualizations")
    print(f"[INFO] Location: {output_dir}/")


if __name__ == "__main__":
    main()
