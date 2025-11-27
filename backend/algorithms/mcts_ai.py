import math
import random
import time
import copy
from typing import List, Tuple
from backend.models.board import Board


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """获取棋子周围distance距离内的空位，空盘返回中心"""
    if board.move_count == 0:
        return [(board.size // 2, board.size // 2)]

    moves = set()
    size = board.size

    existing_stones = []
    for x in range(size):
        for y in range(size):
            if not board.is_empty(x, y):
                existing_stones.append((x, y))

    for (sx, sy) in existing_stones:
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                nx, ny = sx + dx, sy + dy
                if board.is_inside(nx, ny) and board.is_empty(nx, ny):
                    moves.add((nx, ny))

    return list(moves)


class MCTSNode:
    def __init__(self, parent=None, move=None, player_just_moved=None):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.children: List[MCTSNode] = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves: List[Tuple[int, int]] = []

    def uct_select_child(self, exploration_weight=1.414):
        """UCB1公式选择子节点"""
        s = sorted(self.children,
                   key=lambda c: c.wins / c.visits + exploration_weight * math.sqrt(math.log(self.visits) / c.visits))
        return s[-1]

    def add_child(self, move, player_just_moved):
        """添加子节点"""
        n = MCTSNode(parent=self, move=move, player_just_moved=player_just_moved)
        self.children.append(n)
        self.untried_moves.remove(move)
        return n

    def update(self, result):
        """更新访问计数和胜利计数"""
        self.visits += 1
        if self.player_just_moved == result:
            self.wins += 1.0
        elif result == 3:
            self.wins += 0.5


class MCTSAgent:
    def __init__(self, time_limit: float = 2.0, max_iterations: int = 1000):
        self.time_limit = time_limit
        self.max_iterations = max_iterations

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        """MCTS主算法"""
        root = MCTSNode(player_just_moved=3 - player)
        root.untried_moves = get_neighbor_moves(board)

        start_time = time.time()
        count = 0

        while count < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break

            node = root
            sim_board = copy.deepcopy(board)

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                sim_board.place_stone(node.move[0], node.move[1], node.player_just_moved)

            # Expansion
            if node.untried_moves != []:
                m = random.choice(node.untried_moves)
                current_p = 3 - node.player_just_moved
                sim_board.place_stone(m[0], m[1], current_p)
                node = node.add_child(m, current_p)

            # Simulation
            sim_player = 3 - node.player_just_moved
            depth = 0
            while True:
                status = sim_board.get_game_result()
                if status != 0:
                    break

                candidates = get_neighbor_moves(sim_board)
                if not candidates:
                    break

                move = random.choice(candidates)
                sim_board.place_stone(move[0], move[1], sim_player)
                sim_player = 3 - sim_player

                depth += 1
                if depth > 30:
                    break

            # Backpropagation
            final_result = sim_board.get_game_result()
            if final_result == 0:
                final_result = 3

            while node is not None:
                node.update(final_result)
                node = node.parent

            count += 1

        if not root.children:
            moves = get_neighbor_moves(board)
            if moves:
                return random.choice(moves)
            return (0, 0)

        best_child = sorted(root.children, key=lambda c: c.visits)[-1]
        return best_child.move