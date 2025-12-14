import logging
import matplotlib.pyplot as plt
import pandas as pd
import copy
from typing import List, Dict, Any
from backend.engine.board import Board
from backend.analysis.replay import Move

from backend.config.scoring_config import (
    GREEDY_DISTANCE, MINIMAX_DEPTH, MINIMAX_DISTANCE, MINIMAX_CANDIDATE_LIMIT,
    ALPHABETA_DEPTH, ALPHABETA_DISTANCE, ALPHABETA_CANDIDATE_LIMIT,
    MCTS_ITERATION_LIMIT, BRILLIANT_MOVE_THRESHOLD, BLUNDER_MOVE_THRESHOLD,
    CHARTS_DIR, STATS_DIR, CHART_DPI, BOARD_SIZE, LOG_LEVEL, LOG_FORMAT,
    ENABLE_EVALUATION_CACHE, MAX_CACHE_SIZE
)
from backend.utils.scoring_utils import (
    normalize_score_sigmoid, validate_moves_list, classify_move_quality, ensure_directory_exists
)
from backend.ai.basic.classic_ai import (
    GreedyAgent, MinimaxAgent, AlphaBetaAgent, evaluate_board as classic_evaluate_board
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

try:
    from backend.ai.advanced.mcts_ai import MCTSAgent
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE, MCTSAgent = False, None
    logger.warning("MCTS not available")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class MoveScorer:
    """
    Multi-algorithm move scoring system for game replay analysis.
    
    Evaluates each move using multiple AI algorithms:
    - Greedy: fast heuristic-based evaluation
    - Minimax: depth-limited search with heuristic
    - Alpha-Beta: optimized minimax with pruning
    - MCTS: simulation-based Monte Carlo evaluation (optional)
    
    Attributes:
        charts_dir (str): Directory for saving visualization charts
        stats_dir (str): Directory for saving statistics CSV files
        greedy_agent (GreedyAgent): Greedy heuristic evaluator
        minimax_agent (MinimaxAgent): Minimax search evaluator
        alphabeta_agent (AlphaBetaAgent): Alpha-beta pruning evaluator
        mcts_agent (MCTSAgent|None): MCTS evaluator if enabled
        enable_mcts (bool): Whether MCTS evaluation is enabled
    """

    def __init__(self, enable_mcts: bool = False):
        """
        Initialize the move scorer with AI algorithms.
        
        Args:
            enable_mcts (bool): Whether to enable MCTS evaluation (slower but more accurate).
                              Requires 'monte-carlo-tree-search' package to be installed.
                              
        Raises:
            OSError: If output directories cannot be created
        """
        self.charts_dir, self.stats_dir = CHARTS_DIR, STATS_DIR
        ensure_directory_exists(self.charts_dir)
        ensure_directory_exists(self.stats_dir)
        
        self._evaluation_cache = {} if ENABLE_EVALUATION_CACHE else None
        self._cache_hits = self._cache_misses = 0
        
        logger.info("Initializing AI agents...")
        self.greedy_agent = GreedyAgent(distance=GREEDY_DISTANCE)
        self.minimax_agent = MinimaxAgent(
            depth=MINIMAX_DEPTH, distance=MINIMAX_DISTANCE, candidate_limit=MINIMAX_CANDIDATE_LIMIT
        )
        self.alphabeta_agent = AlphaBetaAgent(
            depth=ALPHABETA_DEPTH, distance=ALPHABETA_DISTANCE, candidate_limit=ALPHABETA_CANDIDATE_LIMIT
        )
        logger.info("✓ Classic AI agents initialized")
        
        self.enable_mcts = enable_mcts and MCTS_AVAILABLE
        if self.enable_mcts:
            self.mcts_agent = MCTSAgent(time_limit=None, iteration_limit=MCTS_ITERATION_LIMIT)
            logger.info(f"✓ MCTS initialized (iter={MCTS_ITERATION_LIMIT})")
        else:
            self.mcts_agent = None

    def _evaluate_position(self, board: Board, player: int) -> Dict[str, float]:
        """
        Evaluate a board position using all enabled AI algorithms.
        Each algorithm uses its actual search method to provide independent evaluations.
        
        Args:
            board: Current board state
            player: Player to evaluate for (1 or 2)
            
        Returns:
            Dictionary with scores from each algorithm (normalized to 0-1 range)
        """
        scores = {}
        import math
        
        greedy_score = classic_evaluate_board(board, player)
        scores['greedy'] = 1 / (1 + math.exp(-greedy_score / 1000))
        
        search_board = copy.deepcopy(board)
        try:
            minimax_value, _ = self.minimax_agent._minimax(
                search_board, depth=2, current_player=player, max_player=player
            )
            scores['minimax'] = 1 / (1 + math.exp(-minimax_value / 10000))
        except Exception as e:
            logger.error(f"Minimax error: {e}")
            scores['minimax'] = scores['greedy']
        
        search_board = copy.deepcopy(board)
        try:
            ab_value, _ = self.alphabeta_agent._alphabeta(
                search_board, depth=2, alpha=-math.inf, beta=math.inf,
                current_player=player, max_player=player
            )
            scores['alphabeta'] = 1 / (1 + math.exp(-ab_value / 10000))
        except Exception as e:
            logger.error(f"Alpha-Beta error: {e}")
            scores['alphabeta'] = scores['greedy']
        
        if self.enable_mcts and self.mcts_agent:
            try:
                scores['mcts'] = self.mcts_agent.evaluate_board(board, player)
            except Exception as e:
                logger.error(f"MCTS error: {e}")
                scores['mcts'] = 0.5
        
        return scores

    def score_moves(self, moves: List[Move], game_id: str = "temp") -> Dict[str, Any]:
        """
        Analyze entire game using multi-algorithm scoring.
        
        This method replays the game move-by-move, evaluating each position using
        all enabled AI algorithms to provide comprehensive move quality analysis.
        
        Args:
            moves (List[Move]): List of moves from the game
            game_id (str): Unique identifier for the game (default: "temp")
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - dataframe (pd.DataFrame): Complete move analysis data
                - score_curve (List[float]): Average scores over time
                - critical_moments (List[Dict]): Identified brilliant moves and blunders
                - chart_path (str): Path to main comparison chart
                - chart_paths (Dict[str, str]): Paths to all visualization charts
                - stats_summary (Dict): aggregate statistics
                - csv_path (str): Path to saved CSV file
                
        Example:
            >>> scorer = MoveScorer()
            >>> result = scorer.score_moves(game.moves, game_id="game_123")
            >>> print(result['stats_summary'])
            {'total_moves': 50, 'mean_score': 0.62, ...}
        """
        # Add input validation
        if not moves:
            logger.warning(f"Game {game_id}: No moves to analyze")
            return {
                "error": "No moves to analyze",
                "score_curve": [],
                "critical_moments": [],
                "chart_path": None,
                "chart_paths": {},
                "stats_summary": {},
                "csv_path": None
            }
        
        # Validate moves using utility function
        try:
            validate_moves_list(moves, board_size=BOARD_SIZE)
        except ValueError as e:
            logger.error(f"Game {game_id}: Move validation failed: {e}")
            return {
                "error": str(e),
                "score_curve": [],
                "critical_moments": [],
                "chart_path": None,
                "chart_paths": {},
                "stats_summary": {},
                "csv_path": None
            }
        
        board = Board(size=BOARD_SIZE)
        all_scores = {'greedy': [], 'minimax': [], 'alphabeta': []}
        if self.enable_mcts:
            all_scores['mcts'] = []
        critical_moments = []
        
        algo_names = "Greedy, Minimax, Alpha-Beta" + (", MCTS" if self.enable_mcts else "")
        logger.info(f"🎯 Analyzing game_id={game_id}, moves={len(moves)}, algorithms=[{algo_names}]")
        
        move_iterator = enumerate(moves) if not TQDM_AVAILABLE else tqdm(
            enumerate(moves), total=len(moves), desc=f"分析 {game_id}", unit="步"
        )
        
        for i, move in move_iterator:
            try:
                if not board.place_stone(move.x, move.y, move.player):
                    logger.error(f"Invalid move at step {move.step}: ({move.x}, {move.y})")
                    for algo in all_scores:
                        all_scores[algo].append(0.5)
                    continue
                
                scores = self._evaluate_position(board, move.player)
                for algo, score in scores.items():
                    all_scores[algo].append(score)
                
                avg_score = sum(scores.values()) / len(scores)
                if avg_score >= BRILLIANT_MOVE_THRESHOLD:
                    critical_moments.append({"step": move.step, "type": "Brilliant (妙手)"})
                    logger.debug(f"✨ Brilliant move at step {move.step}: {avg_score:.3f}")
                elif avg_score < BLUNDER_MOVE_THRESHOLD:
                    critical_moments.append({"step": move.step, "type": "Blunder (恶手)"})
                    logger.debug(f"⚠️  Blunder at step {move.step}: {avg_score:.3f}")
                
                if not TQDM_AVAILABLE and ((i + 1) % 10 == 0 or (i + 1) == len(moves)):
                    logger.info(f"Progress: {i + 1}/{len(moves)} moves")
            except Exception as e:
                logger.error(f"Error analyzing move {i+1}: {e}", exc_info=True)
                for algo in all_scores:
                    all_scores[algo].append(0.5)
        
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
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"✓ Saved detailed scores to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            csv_path = None
        
        # Generate visualizations (multiple charts)
        try:
            chart_paths = self.generate_visualizations(df, board, game_id)
            logger.info(f"✓ Generated {len(chart_paths)} visualization charts")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
            chart_paths = {}
        
        # Log completion with summary
        logger.info(f"✅ Analysis complete: mean_score={stats_summary['mean_score']}, "
                   f"brilliant={stats_summary['brilliant_count']}, "
                   f"blunders={stats_summary['blunder_count']}")
        
        return {
            "dataframe": df,
            "score_curve": df['avg_score'].tolist(),
            "critical_moments": critical_moments,
            "chart_path": chart_paths.get('comparison', None),  # Main comparison chart
            "chart_paths": chart_paths,  # All charts
            "stats_summary": stats_summary,
            "csv_path": csv_path
        }

    def generate_visualizations(self, df: pd.DataFrame, board: Board, game_id: str) -> Dict[str, str]:
        """
        Generate comprehensive visualizations including:
        1. Multi-algorithm comparison chart
        2. Move heatmap on board
        3. Score distribution histogram
        
        Args:
            df: DataFrame with all move data
            board: Final board state
            game_id: Game identifier
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        chart_paths = {}
        
        # 1. Multi-algorithm comparison chart (existing)
        chart_paths['comparison'] = self._generate_comparison_chart(df, game_id)
        
        # 2. Move quality heatmap
        chart_paths['heatmap'] = self._generate_move_heatmap(df, board, game_id)
        
        # 3. Score distribution histogram
        chart_paths['distribution'] = self._generate_score_distribution(df, game_id)
        
        return chart_paths

    def _generate_comparison_chart(self, df: pd.DataFrame, game_id: str) -> str:
        """Generate multi-algorithm comparison line chart."""
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
        
        plt.title(f'Multi-Algorithm Move Quality Comparison: {game_id}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Move Step', fontsize=13)
        plt.ylabel('AI Evaluation Score', fontsize=13)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='best', fontsize=11)
        plt.tight_layout()
        
        output_path = os.path.join(self.charts_dir, f"{game_id}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _generate_move_heatmap(self, df: pd.DataFrame, board: Board, game_id: str) -> str:
        """Generate heatmap showing move quality distribution on the board."""
        import numpy as np
        import matplotlib.cm as cm
        
        # Create a heatmap matrix (board_size x board_size)
        board_size = board.size
        heatmap = np.zeros((board_size, board_size))
        heatmap[:] = np.nan  # Use NaN for empty positions
        
        # Fill in scores for each move
        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            heatmap[x][y] = row['avg_score']
        
        # Create the heatmap visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use a colormap that highlights quality (red=bad, yellow=neutral, green=good)
        cmap = cm.get_cmap('RdYlGn')
        im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1, origin='upper')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Move Quality Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add move numbers on the heatmap
        for _, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            player = int(row['player'])
            step = int(row['step'])
            
            # Choose text color based on player
            text_color = 'white' if player == 1 else 'black'
            marker = '●' if player == 1 else '○'
            
            ax.text(y, x, f'{marker}\n{step}', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')
        
        # Set grid
        ax.set_xticks(np.arange(board_size))
        ax.set_yticks(np.arange(board_size))
        ax.set_xticklabels(range(board_size))
        ax.set_yticklabels(range(board_size))
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        
        plt.title(f'Move Quality Heatmap: {game_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Y Coordinate', fontsize=13)
        plt.ylabel('X Coordinate', fontsize=13)
        plt.tight_layout()
        
        output_path = os.path.join(self.charts_dir, f"{game_id}_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _generate_score_distribution(self, df: pd.DataFrame, game_id: str) -> str:
        """Generate histogram showing distribution of move quality scores."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        score_columns = [col for col in df.columns if col.endswith('_score')]
        
        # Plot histogram for each algorithm
        for idx, col in enumerate(score_columns):
            row = idx // 2
            col_idx = idx % 2
            ax = axes[row][col_idx]
            
            algo_name = col.replace('_score', '').title()
            scores = df[col]
            
            # Create histogram
            ax.hist(scores, bins=20, range=(0, 1), alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][idx % 4],
                   edgecolor='black')
            
            # Add vertical lines for mean and median
            mean_val = scores.mean()
            median_val = scores.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                      label=f'Median: {median_val:.3f}')
            
            ax.set_title(f'{algo_name} Score Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Score Distribution Analysis: {game_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = os.path.join(self.charts_dir, f"{game_id}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path