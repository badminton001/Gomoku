"""
MCTS (Monte Carlo Tree Search) Implementation for Gomoku.

This module provides a pure Python implementation of MCTS optimized for Gomoku.
It handles:
- UCB1 selection strategy.
- Simulation with distance-based pruning.
- Backpropagation with correct player perspective.
"""
import copy
import math
import random
from typing import List, Tuple, Optional
from backend.engine.board import Board


def get_neighbor_moves(board: Board, distance: int = 2) -> List[Tuple[int, int]]:
    """Return a list of empty coordinates within 'distance' of existing stones."""
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
    """Represents a node in the MCTS tree."""
    
    def __init__(self, board: Board, player: int, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        
        self.visits = 0
        self.wins = 0.0
        
        self.children: List[MCTSNode] = []
        self.untried_moves: List[Tuple[int, int]] = []
        
        self._is_terminal = None
        self._winner = None
        
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            result = self.board.get_game_result()
            self._is_terminal = (result != 0)
            self._winner = result
        return self._is_terminal
    
    def get_winner(self) -> int:
        if self._winner is None:
            self._winner = self.board.get_game_result()
        return self._winner
    
    def get_untried_moves(self) -> List[Tuple[int, int]]:
        if not self.untried_moves:
            candidates = get_neighbor_moves(self.board, distance=2)
            self.untried_moves = [
                m for m in candidates 
                if self.board.is_inside(m[0], m[1]) and self.board.is_empty(m[0], m[1])
            ]
            random.shuffle(self.untried_moves)
        return self.untried_moves
    
    def select_child_ucb(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            win_rate = child.wins / child.visits
            exploration_term = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = win_rate + exploration_term
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """Expand one untried move."""
        untried = self.get_untried_moves()
        if not untried:
            return None
        
        move = untried.pop()
        new_board = copy.deepcopy(self.board)
        
        if not new_board.place_stone(move[0], move[1], self.player):
            # If move is invalid/forbidden, try next
            if untried:
                return self.expand()
            else:
                return None
        
        next_player = 3 - self.player
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child)
        return child
    
    def simulate(self) -> int:
        """Run a random simulation from this state until invalid or end."""
        sim_board = copy.deepcopy(self.board)
        sim_player = self.player
        max_moves = 225
        
        for _ in range(max_moves):
            result = sim_board.get_game_result()
            if result != 0:
                return result
            
            moves = get_neighbor_moves(sim_board, distance=2)
            if not moves:
                return 3
            
            random.shuffle(moves)
            moved = False
            for move in moves:
                if sim_board.place_stone(move[0], move[1], sim_player):
                    sim_player = 3 - sim_player
                    moved = True
                    break
            
            if not moved:
                return 3
        
        return 3
    
    def backpropagate(self, winner: int):
        """Update node statistics based on simulation result."""
        node = self
        while node is not None:
            node.visits += 1
            
            # Logic: If winner matches the player who made the move to reach this node (parent.player), it's a win.
            if node.parent is None:
                if winner == 3:
                    node.wins += 0.5
                elif winner == node.player:
                    node.wins += 1.0
            else:
                parent_player = 3 - node.player
                if winner == 3:
                    node.wins += 0.5
                elif winner == parent_player:
                    node.wins += 1.0
            
            node = node.parent


class MCTSAgent:
    """Agent using Monte Carlo Tree Search."""
    
    def __init__(self, iteration_limit: int = 300, time_limit: int = None, 
                 exploration_weight: float = 1.414):
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
    
    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        if board.move_count == 0:
            return (board.size // 2, board.size // 2)
        
        root = MCTSNode(copy.deepcopy(board), player)
        
        for _ in range(self.iteration_limit):
            node = root
            
            # Selection
            while node.children and not node.is_terminal():
                node = node.select_child_ucb(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal():
                if node.get_untried_moves():
                    node = node.expand()
                    if node is None:
                        node = root.children[-1] if root.children else root
            
            # Simulation
            if node.is_terminal():
                winner = node.get_winner()
            else:
                winner = node.simulate()
            
            # Backpropagation
            node.backpropagate(winner)
        
        if not root.children:
            candidates = get_neighbor_moves(board, distance=2)
            valid_moves = [m for m in candidates if board.is_valid_move(m[0], m[1])]
            return random.choice(valid_moves) if valid_moves else (-1, -1)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move if best_child.move else (-1, -1)
    
    def evaluate_board(self, board: Board, player: int) -> float:
        """
        Evaluate board position using MCTS simulations.
        
        Args:
            board: Current board state  
            player: Player to evaluate for (1 or 2)
            
        Returns:
            float: Win rate estimate in [0, 1]
        """
        root = MCTSNode(copy.deepcopy(board), player)
        
        for _ in range(self.iteration_limit):
            node = root
            
            # Selection
            while not node.is_terminal() and not node.get_untried_moves() and node.children:
                node = node.select_child_ucb(self.exploration_weight)
            
            # Expansion  
            if not node.is_terminal() and node.get_untried_moves():
                node = node.expand()
                if node is None:
                    node = root if not root.children else root.children[-1]
            
            # Simulation
            winner = node.simulate()
            
            # Backpropagation
            node.backpropagate(winner)
        
        if root.visits == 0:
            return 0.5
            
        return max(0.0, min(1.0, root.wins / root.visits))

