import copy
import random
from typing import List, Tuple

from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS

from backend.models.board import Board


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
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
                        if board_map[nx][ny] == 0 and board.is_valid_move(nx, ny):
                            moves.add((nx, ny))

    return list(moves)


class GomokuState(BaseState):
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
        new_board.place_stone(action, action, self.current_player)
        next_player = 3 - self.current_player
        return GomokuState(new_board, next_player, last_move=action)

    def is_terminal(self) -> bool:
        return self._game_result != 0

    def get_reward(self) -> float:
        if self._game_result == 3:
            return 0.0
        return 1.0 if self._game_result == (3 - self.current_player) else -1.0


class MCTSAgent:
    def __init__(self, time_limit: int = 2000, iteration_limit: int = 1000, **kwargs):
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        if board.move_count == 0:
            return (board.size // 2, board.size // 2)

        search_board = copy.deepcopy(board)
        initial_state = GomokuState(search_board, player)

        searcher = MCTS(
            time_limit=self.time_limit,
            iteration_limit=self.iteration_limit,
            exploration_constant=1.414
        )

        best_action = None
        try:
            best_action = searcher.search(initial_state=initial_state)
        except Exception as e:
            print(f"[MCTS] Search error: {e}")
            best_action = None

        if best_action is None:
            candidates = get_neighbor_moves(board)
            if candidates:
                return random.choice(candidates)
            else:
                return (-1, -1)

        return best_action

    def evaluate_board(self, board: Board, player: int) -> float:
        current_result = board.get_game_result()
        if current_result != 0:
            if current_result == 3:
                return 0.5
            return 1.0 if current_result == player else 0.0

        n_simulations = 30
        wins = 0

        for _ in range(n_simulations):
            sim_board = copy.deepcopy(board)
            current_turn_player = 1 if sim_board.move_count % 2 == 0 else 2
            sim_turn = current_turn_player

            while True:
                res = sim_board.get_game_result()
                if res != 0:
                    if res == player:
                        wins += 1
                    elif res == 3:
                        wins += 0.5
                    break

                moves = get_neighbor_moves(sim_board, distance=1)
                if not moves:
                    moves = get_neighbor_moves(sim_board, distance=2)

                if not moves:
                    break

                action = random.choice(moves)
                sim_board.place_stone(action, action, sim_turn)
                sim_turn = 3 - sim_turn

        return wins / n_simulations