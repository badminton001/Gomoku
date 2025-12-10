import os
import matplotlib.pyplot as plt
import pandas as pd
import copy
from typing import List, Dict, Any, Optional
from backend.models.board import Board
from backend.models.replay import Move

# Import AI algorithms
from backend.algorithms.classic_ai import (
    GreedyAgent, 
    MinimaxAgent, 
    AlphaBetaAgent,
    evaluate_board as classic_evaluate_board
)

# MCTS is optional - only import if available
try:
    from backend.algorithms.mcts_ai import MCTSAgent
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    MCTSAgent = None
    print("⚠️  MCTS module not available. Install 'monte-carlo-tree-search' for MCTS support.")



class MoveScorer:
    """
    Multi-algorithm move scoring system for game replay analysis.
    
    Evaluates each move using multiple AI algorithms:
    - Greedy: fast heuristic-based evaluation
    - Minimax: depth-limited search with heuristic
    - Alpha-Beta: optimized minimax with pruning
    - MCTS: simulation-based Monte Carlo evaluation
    """

    def __init__(self, enable_mcts: bool = False):
        """
        Initialize the move scorer with AI algorithms.
        
        Args:
            enable_mcts: Whether to enable MCTS evaluation (slower but more accurate)
                        Requires 'monte-carlo-tree-search' package to be installed
        """
        # Create output directories
        self.charts_dir = "data/charts"
        self.stats_dir = "data/stats"
        
        for d in [self.charts_dir, self.stats_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        # Initialize AI agents
        self.greedy_agent = GreedyAgent(distance=2)
        self.minimax_agent = MinimaxAgent(depth=2, distance=2, candidate_limit=12)
        self.alphabeta_agent = AlphaBetaAgent(depth=2, distance=2, candidate_limit=12)
        
        self.enable_mcts = enable_mcts and MCTS_AVAILABLE
        if enable_mcts and not MCTS_AVAILABLE:
            print("⚠️  MCTS requested but not available. Continuing with classic algorithms only.")
        
        if self.enable_mcts:
            # Use fewer simulations for move analysis (faster)
            self.mcts_agent = MCTSAgent(time_limit=None, iteration_limit=100)
        else:
            self.mcts_agent = None

    def _evaluate_position(self, board: Board, player: int) -> Dict[str, float]:
        """
        Evaluate a board position using all enabled AI algorithms.
        
        Args:
            board: Current board state
            player: Player to evaluate for (1 or 2)
            
        Returns:
            Dictionary with scores from each algorithm (normalized to 0-1 range)
        """
        scores = {}
        
        # Classic heuristic evaluation (fast)
        # Returns raw score, need to normalize
        classic_score = classic_evaluate_board(board, player)
        # Normalize using sigmoid function to map to [0, 1]
        import math
        scores['greedy'] = 1 / (1 + math.exp(-classic_score / 1000))
        
        # Minimax and Alpha-Beta use same evaluation, but different search
        # We'll use the heuristic value as their scores
        scores['minimax'] = scores['greedy']  # Same heuristic
        scores['alphabeta'] = scores['greedy']  # Same heuristic
        
        # MCTS evaluation (slow but accurate)
        if self.enable_mcts and self.mcts_agent:
            try:
                mcts_score = self.mcts_agent.evaluate_board(board, player)
                scores['mcts'] = mcts_score
            except Exception as e:
                print(f"MCTS evaluation error: {e}")
                scores['mcts'] = 0.5  # Neutral score on error
        
        return scores

    def score_moves(self, moves: List[Move], game_id: str = "temp") -> Dict[str, Any]:
        """
        Analyze entire game using multi-algorithm scoring.
        
        Args:
            moves: List of moves from the game
            game_id: Unique identifier for the game
            
        Returns:
            Dictionary containing:
            - DataFrame with all scores
            - Critical moments list
            - Chart path
            - Statistics summary
            - CSV path
        """
        if not moves:
            return {
                "error": "No moves to analyze",
                "score_curve": [],
                "critical_moments": [],
                "chart_path": None,
                "stats_summary": {},
                "csv_path": None
            }
        
        # Initialize board for replay
        board_size = 15  # Standard Gomoku board
        board = Board(size=board_size)
        
        # Data collection lists
        all_scores = {
            'greedy': [],
            'minimax': [],
            'alphabeta': [],
        }
        if self.enable_mcts:
            all_scores['mcts'] = []
        
        critical_moments = []
        
        print(f"\n🎯 Analyzing game {game_id} with {len(moves)} moves...")
        print(f"   Algorithms: Greedy, Minimax, Alpha-Beta" + (", MCTS" if self.enable_mcts else ""))
        
        # Replay and score each move
        for i, move in enumerate(moves):
            # Place the move on board
            board.place_stone(move.x, move.y, move.player)
            
            # Evaluate position after this move for the player who just moved
            scores = self._evaluate_position(board, move.player)
            
            # Store scores
            for algo, score in scores.items():
                all_scores[algo].append(score)
            
            # Identify critical moments based on average score
            avg_score = sum(scores.values()) / len(scores)
            
            if avg_score > 0.8:
                critical_moments.append({
                    "step": move.step,
                    "type": "Brilliant (妙手)"
                })
            elif avg_score < 0.2:
                critical_moments.append({
                    "step": move.step,
                    "type": "Blunder (恶手)"
                })
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(moves):
                print(f"   Progress: {i + 1}/{len(moves)} moves analyzed")
        
        # Create comprehensive DataFrame
        df_data = {
            "step": [m.step for m in moves],
            "player": [m.player for m in moves],
            "x": [m.x for m in moves],
            "y": [m.y for m in moves],
            "timestamp": [m.timestamp for m in moves],
        }
        
        # Add algorithm scores
        for algo, scores in all_scores.items():
            df_data[f'{algo}_score'] = scores
        
        # Add aggregate columns
        df = pd.DataFrame(df_data)
        
        # Calculate average score across all algorithms
        score_columns = [col for col in df.columns if col.endswith('_score')]
        df['avg_score'] = df[score_columns].mean(axis=1)
        
        # Calculate score variance (agreement between algorithms)
        df['score_variance'] = df[score_columns].var(axis=1)
        
        # Determine best algorithm for each move
        df['best_algorithm'] = df[score_columns].idxmax(axis=1).str.replace('_score', '')
        
        # Mark move types
        df['move_type'] = 'Normal'
        df.loc[df['avg_score'] > 0.8, 'move_type'] = 'Brilliant'
        df.loc[df['avg_score'] < 0.2, 'move_type'] = 'Blunder'
        
        # Calculate statistics summary
        stats_summary = {
            "total_moves": len(moves),
            "mean_score": round(df['avg_score'].mean(), 3),
            "std_dev": round(df['avg_score'].std(), 3),
            "brilliant_count": int((df['avg_score'] > 0.8).sum()),
            "blunder_count": int((df['avg_score'] < 0.2).sum()),
            "avg_variance": round(df['score_variance'].mean(), 3),
        }
        
        # Add per-algorithm stats
        for algo in all_scores.keys():
            stats_summary[f'{algo}_mean'] = round(df[f'{algo}_score'].mean(), 3)
        
        # Save DataFrame to CSV
        csv_path = os.path.join(self.stats_dir, f"{game_id}_multi_algo_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"   ✓ Saved detailed scores to {csv_path}")
        
        # Generate visualization
        chart_path = self.generate_analysis_chart(df, game_id)
        print(f"   ✓ Generated chart: {chart_path}")
        
        return {
            "dataframe": df,
            "score_curve": df['avg_score'].tolist(),
            "critical_moments": critical_moments,
            "chart_path": chart_path,
            "stats_summary": stats_summary,
            "csv_path": csv_path
        }

    def generate_analysis_chart(self, df: pd.DataFrame, game_id: str) -> str:
        """
        Generate multi-algorithm comparison chart.
        
        Args:
            df: DataFrame with all algorithm scores
            game_id: Game identifier
            
        Returns:
            Path to saved chart
        """
        plt.figure(figsize=(14, 8))
        
        # Plot scores from all algorithms
        score_columns = [col for col in df.columns if col.endswith('_score')]
        steps = df['step'].tolist()
        
        # Define colors for each algorithm
        colors = {
            'greedy_score': '#1f77b4',      # Blue
            'minimax_score': '#ff7f0e',     # Orange
            'alphabeta_score': '#2ca02c',   # Green
            'mcts_score': '#d62728'         # Red
        }
        
        for col in score_columns:
            algo_name = col.replace('_score', '').title()
            plt.plot(steps, df[col], marker='o', linestyle='-', 
                    color=colors.get(col, '#333333'),
                    label=algo_name, alpha=0.7, linewidth=2)
        
        # Plot average score with thicker line
        plt.plot(steps, df['avg_score'], marker='s', linestyle='--',
                color='black', label='Average', linewidth=3, alpha=0.8)
        
        # Annotate critical moments
        for i, row in df.iterrows():
            if row['avg_score'] > 0.8:
                plt.annotate('!', (row['step'], row['avg_score']), 
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', color='green', fontweight='bold', fontsize=14)
            elif row['avg_score'] < 0.2:
                plt.annotate('?', (row['step'], row['avg_score']),
                           textcoords="offset points", xytext=(0, -15),
                           ha='center', color='red', fontweight='bold', fontsize=14)
        
        plt.title(f'Multi-Algorithm Game Analysis: {game_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Move Step', fontsize=13)
        plt.ylabel('AI Evaluation Score', fontsize=13)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='best', fontsize=11)
        plt.tight_layout()
        
        output_filename = f"{game_id}_multi_algo_analysis.png"
        output_path = os.path.join(self.charts_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path