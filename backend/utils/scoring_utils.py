"""
Utility functions for game scoring and analysis.

This module provides helper functions for the move scoring system including
score normalization, validation, and statistical operations.

Author: Person B - Game Replay & Scoring Team
"""

import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from backend.analysis.replay import Move
from backend.config.scoring_config import (
    SIGMOID_SCALE_GREEDY,
    SIGMOID_SCALE_SEARCH,
    MIN_COORDINATE,
    MAX_COORDINATE,
    MIN_MOVES_FOR_ANALYSIS,
    MAX_MOVES_FOR_ANALYSIS,
    BRILLIANT_MOVE_THRESHOLD,
    BLUNDER_MOVE_THRESHOLD
)

logger = logging.getLogger(__name__)


def normalize_score_sigmoid(raw_score: float, scale: float = SIGMOID_SCALE_SEARCH) -> float:
    """
    Normalize raw evaluation score to [0, 1] range using sigmoid function.
    
    Args:
        raw_score: Raw score from evaluation function
        scale: Scale factor for sigmoid normalization
        
    Returns:
        Normalized score in [0, 1] range
        
    Example:
        >>> normalize_score_sigmoid(10000)
        0.731...
    """
    return 1.0 / (1.0 + math.exp(-raw_score / scale))


def validate_move(move: Move, board_size: int = 15) -> bool:
    """
    Validate a single move.
   
    Args:
        move: Move object to validate
        board_size: Size of the game board
        
    Returns:
        True if valid
        
    Raises:
        TypeError: If move is not a Move object
        ValueError: If move coordinates are out of bounds
    """
    if not isinstance(move, Move):
        raise TypeError(f"Expected Move object, got {type(move)}")
    
    if not (MIN_COORDINATE <= move.x < board_size):
        raise ValueError(f"Invalid x coordinate: {move.x} (must be in [0, {board_size-1}])")
    
    if not (MIN_COORDINATE <= move.y < board_size):
        raise ValueError(f"Invalid y coordinate: {move.y} (must be in [0, {board_size-1}])")
    
    if move.player not in [1, 2]:
        raise ValueError(f"Invalid player: {move.player} (must be 1 or 2)")
    
    return True


def validate_moves_list(moves: List[Move], board_size: int = 15) -> bool:
    """
    Validate a list of moves before processing.
    
    Args:
        moves: List of Move objects
        board_size: Size of the game board
        
    Returns:
        True if all moves are valid
        
    Raises:
        ValueError: If move list is invalid
    """
    if not moves:
        raise ValueError("Move list cannot be empty")
    
    if len(moves) < MIN_MOVES_FOR_ANALYSIS:
        raise ValueError(
            f"Too few moves: {len(moves)} (minimum: {MIN_MOVES_FOR_ANALYSIS})"
        )
    
    if len(moves) > MAX_MOVES_FOR_ANALYSIS:
        logger.warning(
            f"Move list is very long ({len(moves)} moves). "
            f"Processing may take significant time."
        )
    
    # Validate each move
    for i, move in enumerate(moves):
        try:
            validate_move(move, board_size)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid move at index {i}: {e}")
    
    logger.debug(f"Validated {len(moves)} moves successfully")
    return True


def classify_move_quality(score: float) -> str:
    """
    Classify move quality based on score.
    
    Args:
        score: Normalized score in [0, 1] range
        
    Returns:
        Move quality classification ('Brilliant', 'Normal', 'Blunder')
        
    Example:
        >>> classify_move_quality(0.85)
        'Brilliant'
        >>> classify_move_quality(0.5)
        'Normal'
        >>> classify_move_quality(0.15)
        'Blunder'
    """
    if score >= BRILLIANT_MOVE_THRESHOLD:
        return 'Brilliant'
    elif score < BLUNDER_MOVE_THRESHOLD:
        return 'Blunder'
    else:
        return 'Normal'


def detect_critical_moments(
    scores: List[float],
    threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Detect critical moments by analyzing score changes.
    
    Args:
        scores: List of scores over time
        threshold: Minimum score change to be considered critical
        
    Returns:
        List of dictionaries with critical moment information
        
    Example:
        >>> scores = [0.5, 0.5, 0.7, 0.3, 0.8]
        >>> moments = detect_critical_moments(scores, threshold=0.15)
        >>> len(moments)
        2
    """
    critical_moments = []
    
    for i in range(1, len(scores)):
        score_change = abs(scores[i] - scores[i-1])
        
        if score_change >= threshold:
            critical_moments.append({
                'index': i,
                'previous_score': scores[i-1],
                'current_score': scores[i],
                'change': scores[i] - scores[i-1],
                'magnitude': score_change
            })
    
    return critical_moments


def calculate_score_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of scores.
    
    Args:
        scores: List of scores
        
    Returns:
        Dictionary with statistical metrics
        
    Example:
        >>> stats = calculate_score_statistics([0.5, 0.6, 0.7, 0.8])
        >>> stats['mean']
        0.65
    """
    if not scores:
        return {
            'mean': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0,
            'variance': 0.0
        }
    
    import numpy as np
    
    return {
        'mean': float(np.mean(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'std': float(np.std(scores)),
        'variance': float(np.var(scores)),
        'median': float(np.median(scores))
    }


def format_score_percentage(score: float, decimal_places: int = 2) -> str:
    """
    Format score as percentage string.
    
    Args:
        score: Score in [0, 1] range
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
        
    Example:
        >>> format_score_percentage(0.8523, 2)
        '85.23%'
    """
    return f"{score * 100:.{decimal_places}f}%"


def ensure_directory_exists(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    import os
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return directory


# Module-level test
if __name__ == "__main__":
    print("Scoring Utils Test")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - normalize_score_sigmoid()")
    print("  - validate_move()")
    print("  - validate_moves_list()")
    print("  - classify_move_quality()")
    print("  - detect_critical_moments()")
    print("  - calculate_score_statistics()")
    print("  - format_score_percentage()")
    print("  - ensure_directory_exists()")
