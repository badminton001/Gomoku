import copy
from typing import List, Tuple
from backend.models.board import Board
from mcts.base.base import BaseState
from mcts.searcher.mcts import MCTS


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """获取棋子周围distance范围内的空位"""
    if board.move_count == 0:
        return [(board.size // 2, board.size // 2)]
    moves = set()
    for x in range(board.size):
        for y in range(board.size):
            if not board.is_empty(x, y):
                for dx in range(-distance, distance + 1):
                    for dy in range(-distance, distance + 1):
                        nx, ny = x + dx, y + dy
                        if board.is_valid_move(nx, ny):
                            moves.add((nx, ny))
    return list(moves)


class GomokuState(BaseState):
    """五子棋游戏状态"""
    def __init__(self, board: Board, current_player: int, last_player: int = None):
        self.board = copy.deepcopy(board)
        self.current_player = current_player
        self.last_player = last_player

    def get_current_player(self) -> int:
        """返回当前玩家（1 或 -1）"""
        return 1 if self.current_player == 1 else -1

    def get_possible_actions(self) -> List[Tuple[int, int]]:
        """返回所有合法着法"""
        return get_neighbor_moves(self.board)

    def take_action(self, action: Tuple[int, int]) -> "GomokuState":
        """执行着法，返回新状态"""
        new_board = copy.deepcopy(self.board)
        new_board.place_stone(action, action, self.current_player)
        next_player = 3 - self.current_player
        return GomokuState(new_board, next_player, self.current_player)

    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.board.get_game_result() != 0

    def get_reward(self) -> float:
        """返回奖励值"""
        result = self.board.get_game_result()
        if result == 0:
            return 0.0
        elif result == 3:
            return 0.0
        elif result == self.last_player:
            return 1.0
        else:
            return -1.0


class MCTSAgent:
    """蒙特卡洛树搜索"""
    def __init__(self, time_limit: float = 2.0):
        self.time_limit = time_limit

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        """获取最优着法"""
        initial_state = GomokuState(board, player)
        searcher = MCTS(time_limit=int(self.time_limit * 1000))
        best_action = searcher.search(initial_state=initial_state)
        if best_action is None:
            moves = get_neighbor_moves(board)
            if moves:
                return moves
            return (board.size // 2, board.size // 2)
        return best_action

    def evaluate_board(self, board: Board, player: int = 1) -> float:
        """快速评估棋盘"""
        initial_state = GomokuState(board, player)
        searcher = MCTS(time_limit=100)
        best_action = searcher.search(initial_state=initial_state)
        # 简单评估：如果找到着法就返回 50，否则返回 50
        return 50.0