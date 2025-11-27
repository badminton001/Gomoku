import math
import random
import time
import pickle
import copy
from typing import List, Tuple, Dict, Optional
from board import Board


# 获取候选步：棋盘现有棋子周围distance距离内的空位
def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """
    减少搜索空间：只看棋子附近的空位，空盘则返回中心点。
    """
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


# MCTS 树节点
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
        """UCB1 公式选择子节点"""
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
        """更新节点访问和胜利次数"""
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
        """MCTS 求解"""
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
            return random.choice(get_neighbor_moves(board))

        best_child = sorted(root.children, key=lambda c: c.visits)[-1]
        return best_child.move


# Q-Learning 智能体
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table: Dict[str, float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None

    def get_state_key(self, board: Board) -> str:
        """棋盘转为字符串状态"""
        return board.to_string()

    def get_qa_key(self, state: str, action: Tuple[int, int]) -> str:
        """状态-动作 Key"""
        return f"{state}|{action[0]},{action[1]}"

    def get_q(self, state: str, action: Tuple[int, int]) -> float:
        return self.q_table.get(self.get_qa_key(state, action), 0.0)

    def set_q(self, state: str, action: Tuple[int, int], value: float):
        self.q_table[self.get_qa_key(state, action)] = value

    def get_move(self, board: Board, player: int, training: bool = False) -> Tuple[int, int]:
        """获取动作"""
        moves = get_neighbor_moves(board)
        if not moves:
            return (7, 7)

        state = self.get_state_key(board)

        # Epsilon-Greedy
        if training and random.random() < self.epsilon:
            action = random.choice(moves)
        else:
            random.shuffle(moves)
            best_move = moves[0]
            max_q = -float('inf')

            for move in moves:
                q = self.get_q(state, move)
                if q > max_q:
                    max_q = q
                    best_move = move
            action = best_move

        if training:
            self.last_state = state
            self.last_action = action

        return action

    def learn(self, current_board: Board, reward: float):
        """Q值更新"""
        if self.last_state is None or self.last_action is None:
            return

        current_state = self.get_state_key(current_board)

        moves = get_neighbor_moves(current_board)
        max_next_q = 0.0
        if moves:
            max_next_q = max([self.get_q(current_state, m) for m in moves])

        old_q = self.get_q(self.last_state, self.last_action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.set_q(self.last_state, self.last_action, new_q)

    def save_model(self, filename="q_model.pkl"):
        """保存Q表"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Model saved to {filename}, Size: {len(self.q_table)} entries")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename="q_model.pkl"):
        """加载Q表"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filename}, Size: {len(self.q_table)} entries")
        except FileNotFoundError:
            print("Model file not found, starting with empty Q-table.")
        except Exception as e:
            print(f"Error loading model: {e}")