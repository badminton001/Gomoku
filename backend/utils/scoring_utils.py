"""Utility functions for scoring and analysis."""

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
    """Normalize score to [0, 1] using sigmoid."""
    return 1.0 / (1.0 + math.exp(-raw_score / scale))


def validate_move(move: Move, board_size: int = 15) -> bool:
    """Validate a single move."""
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
    """Validate a list of moves."""
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
    """Classify move ('Brilliant', 'Normal', 'Blunder')."""
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
    """Detect significant score changes."""
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
    """Calculate basic statistics (mean, std, etc.)."""
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
    """Format score as percentage."""
    return f"{score * 100:.{decimal_places}f}%"


def ensure_directory_exists(directory: str) -> str:
    """Create directory if not exists."""
    import os
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return directory


# Module-level test
if __name__ == "__main__":
    print("Scoring Utils Test")

    print("\nAvailable functions:")
    print("  - normalize_score_sigmoid()")
    print("  - validate_move()")
    print("  - validate_moves_list()")
    print("  - classify_move_quality()")
    print("  - detect_critical_moments()")
    print("  - calculate_score_statistics()")
    print("  - format_score_percentage()")
    print("  - ensure_directory_exists()")
