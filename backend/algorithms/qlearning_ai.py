import random
import pickle
from typing import List, Tuple, Dict
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
        """状态-动作Key"""
        return f"{state}|{action[0]},{action[1]}"

    def get_q(self, state: str, action: Tuple[int, int]) -> float:
        """获取Q值，不存在返回0.0"""
        return self.q_table.get(self.get_qa_key(state, action), 0.0)

    def set_q(self, state: str, action: Tuple[int, int], value: float):
        """设置Q值"""
        self.q_table[self.get_qa_key(state, action)] = value

    def get_move(self, board: Board, player: int, training: bool = False) -> Tuple[int, int]:
        """
        选择动作，training=True时使用Epsilon-Greedy进行探索
        """
        moves = get_neighbor_moves(board)
        if not moves:
            return (7, 7)

        state = self.get_state_key(board)

        # Epsilon-Greedy策略
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
        """
        根据当前局面和奖励更新上一步的Q值
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
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