"""
Configuration package for Gomoku backend.

This package contains configuration files for different modules.
"""

from .scoring_config import *

__all__ = [
    # AI Agent Configuration
    'GREEDY_DISTANCE',
    'MINIMAX_DEPTH',
    'MINIMAX_DISTANCE',
    'MINIMAX_CANDIDATE_LIMIT',
    'ALPHABETA_DEPTH',
    'ALPHABETA_DISTANCE',
    'ALPHABETA_CANDIDATE_LIMIT',
    'MCTS_TIME_LIMIT',
    'MCTS_ITERATION_LIMIT',
    
    # Thresholds
    'BRILLIANT_MOVE_THRESHOLD',
    'BLUNDER_MOVE_THRESHOLD',
    'CRITICAL_SCORE_DROP',
    
    # Directories
    'CHARTS_DIR',
    'STATS_DIR',
    'REPLAY_DIR',
    
    # Performance
    'ENABLE_EVALUATION_CACHE',
    'MAX_CACHE_SIZE',
    
    # Board
    'BOARD_SIZE',
    'WIN_LENGTH',
]
