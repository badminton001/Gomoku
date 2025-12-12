"""
MCTS (Monte Carlo Tree Search) Implementation for Gomoku
完全重写的MCTS实现，不依赖第三方库

Key features:
- Correct reward propagation  
- Proper forbidden move handling
- Optimized for Gomoku with forbidden move rules
"""
import copy
import math
import random
from typing import List, Tuple, Optional
from backend.models.board import Board


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """获取棋盘上已有棋子周围的空位
    
    Args:
        board: 棋盘对象
        distance: 搜索距离
        
    Returns:
        空位坐标列表
    """
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


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, board: Board, player: int, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        """
        Args:
            board: 当前棋盘状态
            player: 当前要下棋的玩家 (1 or 2)
            parent: 父节点
            move: 从父节点到此节点的走法
        """
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        
        # MCTS统计信息
        self.visits = 0
        self.wins = 0.0  # 使用float以支持平局的0.5
        
        # 子节点
        self.children: List[MCTSNode] = []
        self.untried_moves: List[Tuple[int, int]] = []
        
        # 是否是终止节点
        self._is_terminal = None
        self._winner = None
        
    def is_terminal(self) -> bool:
        """检查是否是终止状态"""
        if self._is_terminal is None:
            result = self.board.get_game_result()
            self._is_terminal = (result != 0)
            self._winner = result
        return self._is_terminal
    
    def get_winner(self) -> int:
        """获取胜者 (0=进行中, 1/2=玩家, 3=平局)"""
        if self._winner is None:
            self._winner = self.board.get_game_result()
        return self._winner
    
    def get_untried_moves(self) -> List[Tuple[int, int]]:
        """获取未尝试的走法"""
        if not self.untried_moves:
            # 第一次调用时初始化
            candidates = get_neighbor_moves(self.board, distance=2)
            # 过滤掉明显不合法的走法（但不做昂贵的禁手检查）
            self.untried_moves = [
                m for m in candidates 
                if self.board.is_inside(m[0], m[1]) and self.board.is_empty(m[0], m[1])
            ]
            random.shuffle(self.untried_moves)  # 随机化顺序
        return self.untried_moves
    
    def select_child_ucb(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """使用UCB1公式选择最佳子节点
        
        UCB1 = win_rate + C * sqrt(ln(parent_visits) / child_visits)
        """
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # 未访问的节点给予最高优先级
                return child
            
            # 计算UCB值
            win_rate = child.wins / child.visits
            exploration_term = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = win_rate + exploration_term
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """扩展一个新的子节点"""
        untried = self.get_untried_moves()
        
        if not untried:
            return None
        
        # 选择一个未尝试的走法
        move = untried.pop()
        
        # 创建新的棋盘状态
        new_board = copy.deepcopy(self.board)
        success = new_board.place_stone(move[0], move[1], self.player)
        
        if not success:
            # 这是禁手，不展开，继续尝试其他走法
            if untried:
                return self.expand()
            else:
                return None
        
        # 创建子节点
        next_player = 3 - self.player
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child)
        
        return child
    
    def simulate(self) -> int:
        """从当前状态模拟到游戏结束
        
        Returns:
            胜者 (1, 2, 或 3表示平局)
        """
        sim_board = copy.deepcopy(self.board)
        sim_player = self.player
        max_moves = 225  # 15x15最多225步
        
        for _ in range(max_moves):
            # 检查游戏是否结束
            result = sim_board.get_game_result()
            if result != 0:
                return result
            
            # 获取可能的走法
            moves = get_neighbor_moves(sim_board, distance=2)
            if not moves:
                return 3  # 平局
            
            # 随机选择一个合法走法
            random.shuffle(moves)
            moved = False
            for move in moves:
                if sim_board.place_stone(move[0], move[1], sim_player):
                    sim_player = 3 - sim_player
                    moved = True
                    break
            
            if not moved:
                return 3  # 没有合法走法，平局
        
        return 3  # 超过最大步数，平局
    
    def backpropagate(self, winner: int):
        """回传模拟结果
        
        Args:
            winner: 胜者 (1, 2, 或 3表示平局)
        """
        node = self
        while node is not None:
            node.visits += 1
            
            # CRITICAL FIX: 奖励应该从父节点玩家的视角
            # node.player是刚刚走完这步的玩家
            # 我们需要从父节点（之前的玩家）视角评估
            # 
            # 举例：如果这个节点是player=1刚下的
            # 那么parent.player=2（之前的玩家）
            # 如果winner=1，说明parent的对手赢了，对parent不利
            #
            # 但在回传时，我们在当前节点更新统计
            # node.wins应该表示"从node.player视角的胜率"
            # 如果winner == node.player，说明node的玩家赢了，这对之前选择这个节点的决策不利
            # 因为node.player是对手
            #
            # 等等！我搞混了。让我重新理清：
            # - 节点存储的是"下完这步后的状态" 
            # - node.player是"接下来要下的玩家"（不是刚下完的）
            # - parent选择这个节点，意味着parent.player下了这步棋
            # - node.wins统计应该是"对parent有利的次数"
            # - 如果winner == parent.player，则parent赢了，wins+1
            # - 如果winner == node.player，则对手赢了，wins+0
            #
            # 但是我们在node上更新，不知道parent.player...
            # 简单方法：parent.player = 3 - node.player
            
            if node.parent is None:
                # 根节点，从根节点玩家视角
                if winner == 3:
                    node.wins += 0.5
                elif winner == node.player:
                    node.wins += 1.0
                else:
                    node.wins += 0.0
            else:
                # 非根节点
                # parent.player下了走法到达node
                # parent.player = 3 - node.player
                parent_player = 3 - node.player
                
                if winner == 3:
                    node.wins += 0.5
                elif winner == parent_player:
                    # parent的玩家赢了，这个走法好
                    node.wins += 1.0
                else:
                    # 对手赢了，这个走法不好
                    node.wins += 0.0
            
            node = node.parent


class MCTSAgent:
    """MCTS智能体"""
    
    def __init__(self, iteration_limit: int = 300, time_limit: int = None, 
                 exploration_weight: float = 1.414):
        """
        Args:
            iteration_limit: 最大迭代次数 (默认300，平衡性能和速度)
            time_limit: 时间限制（毫秒，暂未实现）
            exploration_weight: UCB探索权重
        """
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
    
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        """获取最佳走法
        
        Args:
            board: 当前棋盘
            player: 当前玩家
            
        Returns:
            最佳走法 (x, y)
        """
        # 第一步直接下天元
        if board.move_count == 0:
            return (board.size // 2, board.size // 2)
        
        # 创建根节点
        root = MCTSNode(copy.deepcopy(board), player)
        
        # MCTS主循环
        for iteration in range(self.iteration_limit):
            node = root
            
            # 1. Selection - 选择到叶子节点
            while node.children and not node.is_terminal():
                node = node.select_child_ucb(self.exploration_weight)
            
            # 2. Expansion - 如果不是终止节点，扩展一个子节点
            if not node.is_terminal():
                if node.get_untried_moves():
                    node = node.expand()
                    if node is None:
                        # 扩展失败（可能都是禁手），用当前节点模拟
                        node = root.children[-1] if root.children else root
            
            # 3. Simulation - 模拟到游戏结束
            if node.is_terminal():
                winner = node.get_winner()
            else:
                winner = node.simulate()
            
            # 4. Backpropagation - 回传结果
            node.backpropagate(winner)
        
        # 选择访问次数最多的子节点
        if not root.children:
            # 没有子节点，随机选择一个合法走法
            candidates = get_neighbor_moves(board, distance=2)
            valid_moves = [m for m in candidates if board.is_valid_move(m[0], m[1])]
            if valid_moves:
                return random.choice(valid_moves)
            return (-1, -1)
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        # 验证选择的走法是否合法
        if best_child.move and board.is_valid_move(best_child.move[0], best_child.move[1]):
            return best_child.move
        
        # 如果最佳走法不合法（禁手），选择其他合法走法
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.move and board.is_valid_move(child.move[0], child.move[1]):
                return child.move
        
        # 所有MCTS选择都是禁手，fallback到随机合法走法
        candidates = get_neighbor_moves(board, distance=2)
        valid_moves = [m for m in candidates if board.is_valid_move(m[0], m[1])]
        if valid_moves:
            return random.choice(valid_moves)
        
        return (-1, -1)
