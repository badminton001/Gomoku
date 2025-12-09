import copy
import random
from typing import List, Tuple

from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS

from backend.models.board import Board


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """获取棋子周围distance范围内的空位"""
    if board.move_count == 0:
        return [(board.size // 2, board.size // 2)]
    
    moves = set()
    size = board.size
    board_map = board.board
    
    for x in range(size):
        for y in range(size):
            if board_map[x][y] != 0:
                x_min = max(0, x - distance)
                x_max = min(size, x + distance + 1)
                y_min = max(0, y - distance)
                y_max = min(size, y + distance + 1)
                
                for nx in range(x_min, x_max):
                    for ny in range(y_min, y_max):
                        if board_map[nx][ny] == 0:
                            moves.add((nx, ny))
    
    return list(moves)


class GomokuState(BaseState):
    """MCTS库的State接口适配"""
    
    def __init__(self, board: Board, current_player: int, last_move: Tuple[int, int] = None):
        self.board = board
        self.current_player = current_player
        self.last_move = last_move
        self._game_result = self.board.get_game_result()

    def get_current_player(self) -> int:
        return self.current_player

    def get_possible_actions(self) -> List[Tuple[int, int]]:
        return get_neighbor_moves(self.board, distance=2)

    def take_action(self, action: Tuple[int, int]) -> "GomokuState":
        new_board = copy.deepcopy(self.board)
        new_board.place_stone(action[0], action[1], self.current_player)
        next_player = 3 - self.current_player
        return GomokuState(new_board, next_player, last_move=action)

    def is_terminal(self) -> bool:
        return self._game_result != 0

    def get_reward(self) -> float:
        if self._game_result == 3:
            return 0.0
        return 1.0 if self._game_result == (3 - self.current_player) else -1.0


class MCTSAgent:
    """Monte Carlo Tree Search AI"""
    
    def __init__(self, iteration_limit: int = 1000, time_limit: int = None, **kwargs):
        """Initialize MCTS agent
        
        Args:
            iteration_limit: Number of MCTS iterations (recommended: 500-2000)
            time_limit: Time limit in ms (not used with iteration_limit)
        
        Note: MCTS library doesn't support both limits simultaneously.
              We prioritize iteration_limit for consistent behavior.
        """
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit if iteration_limit is None else None

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        """Get best move using MCTS"""
        # First move: return center
        if board.move_count == 0:
            return (board.size // 2, board.size // 2)

        search_board = copy.deepcopy(board)
        initial_state = GomokuState(search_board, player)
        
        # Create MCTS searcher with only one limit type
        if self.iteration_limit is not None:
            searcher = MCTS(
                iteration_limit=self.iteration_limit,
                exploration_constant=1.414
            )
        else:
            searcher = MCTS(
                time_limit=self.time_limit,
                exploration_constant=1.414
            )

        try:
            best_action = searcher.search(initial_state=initial_state)
        except Exception as e:
            print(f"[MCTS] Search error: {e}")
            best_action = None

        # Fallback to random valid move if search failed
        if best_action is None:
            candidates = get_neighbor_moves(board)
            if candidates:
                return random.choice(candidates)
            return (board.size // 2, board.size // 2)
            
        return best_action

    def evaluate_board(self, board: Board, player: int) -> float:
        """Quick board evaluation (not used in MCTS search)"""
        return 50.0
