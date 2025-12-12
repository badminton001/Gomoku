"""
Scoring configuration for Gomoku game replay analysis.

This module centralizes all configuration constants for the move scoring system,
including AI agent parameters, evaluation thresholds, and output settings.

Author: Person B - Game Replay & Scoring Team
"""

# ==================== AI Agent Configuration ====================

# Greedy Agent
GREEDY_DISTANCE = 2  # Search distance for candidate moves

# Minimax Agent
MINIMAX_DEPTH = 2  # Search depth (higher = slower but more accurate)
MINIMAX_DISTANCE = 2  # Distance for candidate move generation
MINIMAX_CANDIDATE_LIMIT = 12  # Maximum candidates to consider per position

# Alpha-Beta Agent
ALPHABETA_DEPTH = 2  # Search depth with pruning
ALPHABETA_DISTANCE = 2  # Distance for candidate moves
ALPHABETA_CANDIDATE_LIMIT = 12  # Maximum candidates

# MCTS Agent
MCTS_TIME_LIMIT = None  # Time limit in seconds (None = use iteration limit)
MCTS_ITERATION_LIMIT = 100  # Number of MCTS simulations per move
MCTS_EXPLORATION_CONSTANT = 1.414  # UCT exploration parameter (√2)

# ==================== Scoring Thresholds ====================

# Move quality classification
BRILLIANT_MOVE_THRESHOLD = 0.8  # Score >= this = brilliant move (妙手)
BLUNDER_MOVE_THRESHOLD = 0.2  # Score < this = blunder (恶手)

# Critical moment detection
CRITICAL_SCORE_DROP = 0.15  # Drop >= this between moves = critical moment
SIGNIFICANT_SCORE_CHANGE = 0.10  # Threshold for significant changes

# ==================== Score Normalization ====================

# Sigmoid normalization parameters
SIGMOID_SCALE_GREEDY = 1000  # Scale factor for greedy scores
SIGMOID_SCALE_SEARCH = 10000  # Scale factor for search algorithm scores

# ==================== Output Settings ====================

# Directory structure
CHARTS_DIR = "data/charts"  # Directory for visualization charts
STATS_DIR = "data/stats"  # Directory for statistics CSV files
REPLAY_DIR = "data/replays"  # Directory for game replay files

# File naming
CHART_COMPARISON_SUFFIX = "comparison"  # Main comparison chart suffix
CHART_HEATMAP_SUFFIX = "heatmap"  # Heatmap chart suffix
CHART_DISTRIBUTION_SUFFIX = "distribution"  # Score distribution chart suffix
STATS_CSV_SUFFIX = "multi_algo_stats"  # Statistics CSV suffix

# ==================== Visualization Settings ====================

# Chart dimensions
CHART_WIDTH = 14  # Figure width in inches
CHART_HEIGHT = 8  # Figure height in inches
CHART_DPI = 300  # Resolution for saved charts

# Color scheme
COLOR_GREEDY = '#1f77b4'  # Blue
COLOR_MINIMAX = '#ff7f0e'  # Orange
COLOR_ALPHABETA = '#2ca02c'  # Green
COLOR_MCTS = '#d62728'  # Red
COLOR_AVERAGE = '#000000'  # Black

# Heatmap colors
HEATMAP_COLORMAP = 'RdYlGn'  # Red-Yellow-Green scale
HEATMAP_BOARD_COLOR = '#f0f0f0'  # Light gray for empty cells

# ==================== Performance Settings ====================

# Caching
ENABLE_EVALUATION_CACHE = True  # Whether to cache board evaluations
MAX_CACHE_SIZE = 10000  # Maximum number of cached evaluations

# Batch processing
BATCH_SIZE = 50  # Number of moves to process before checkpointing

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_PROGRESS_INTERVAL = 10  # Log progress every N moves (if no tqdm)

# ==================== Board Settings ====================

# Standard Gomoku
BOARD_SIZE = 15  # 15x15 board
WIN_LENGTH = 5  # 5 consecutive stones to win

# ==================== Validation Settings ====================

# Input validation
VALIDATE_MOVES = True  # Whether to validate move list before processing
MIN_MOVES_FOR_ANALYSIS = 1  # Minimum number of moves required
MAX_MOVES_FOR_ANALYSIS = 500  # Maximum moves (prevent excessive processing)

# Move coordinate validation
MIN_COORDINATE = 0
MAX_COORDINATE = BOARD_SIZE - 1

# ==================== Export Settings ====================

# CSV export
CSV_DECIMAL_PLACES = 4  # Decimal places for scores in CSV
CSV_ENCODING = 'utf-8'  # Character encoding for CSV files

# JSON export
JSON_INDENT = 2  # Indentation for JSON files
JSON_ENCODING = 'utf-8'  # Character encoding for JSON files
