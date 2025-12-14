import time
import math
import random
from typing import Tuple, List, Optional
from backend.engine.board import Board

# -------------------------------------------------------------------------
# Gomoku Shape Scoring (Heuristic)
# -------------------------------------------------------------------------
# We define scores for different patterns. 
# "Live" means open on both ends. "Dead" means blocked on one end.
# 5 is win.
SCORE_FIVE = 10000000
SCORE_LIVE_4 = 1000000
SCORE_DEAD_4 = 100000 # Still dangerous, forces defense
SCORE_LIVE_3 = 100000 # Requires defense
SCORE_DEAD_3 = 1000  # Good for developing
SCORE_LIVE_2 = 100
SCORE_DEAD_2 = 10

class AlphaBetaAgent:
    """
    A Strong Rule-Based AI using Alpha-Beta Pruning with Iterative Deepening.
    No training required. Ready to play immediately.
    """
    def __init__(self, depth: int = 4, time_limit: float = 4.0):
        self.depth = depth
        self.time_limit = time_limit # Seconds per move
        self.start_time = 0
        self.nodes_explored = 0

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        self.start_time = time.time()
        self.nodes_explored = 0
        
        # If board is empty, play center
        if board.move_count == 0:
            return (board.size // 2, board.size // 2)

        # 1. Iterative Deepening
        # Start with depth 2, then 4... until time runs out
        best_move = (-1, -1)
        
        try:
            # We try search depth 2 first (very fast)
            # If we find a winning line, we stop.
            val, move = self.alpha_beta_search(board, player, depth=2, alpha=-math.inf, beta=math.inf)
            best_move = move
            if val >= SCORE_FIVE: # Found a win
                return best_move
                
            # If time permits, go deeper
            if time.time() - self.start_time < (self.time_limit * 0.3):
                val, move = self.alpha_beta_search(board, player, depth=4, alpha=-math.inf, beta=math.inf)
                best_move = move
                
        except TimeoutError:
            pass # Return best move found so far

        return best_move

    def alpha_beta_search(self, board: Board, player: int, depth: int, alpha: float, beta: float) -> Tuple[float, Tuple[int, int]]:
        # Check timeout
        if (self.nodes_explored & 1023) == 0: # Check every 1024 nodes
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
        self.nodes_explored += 1

        # Leaf or Game Over
        result = board.get_game_result()
        if result != 0:
            if result == player: return SCORE_FIVE * 10, (-1, -1) # Win
            if result == (3 - player): return -SCORE_FIVE * 10, (-1, -1) # Loss
            return 0, (-1, -1) # Draw

        if depth == 0:
            return self.evaluate_board(board, player), (-1, -1)

        # Move Ordering: Essential for Alpha-Beta to work well
        opponent = 3 - player
        candidates = self.get_sorted_moves(board, player)
        
        if not candidates:
            return 0, (-1, -1)

        best_move = candidates[0]
        best_val = -math.inf

        # Maximize for 'player'
        for move in candidates:
            mx, my = move
            board.place_stone(mx, my, player)
            
            # Recursive call (Opponent minimizes)
            # Note: We pass -beta, -alpha and negate result for NegaMax style simplifications, 
            # but here I write explicit MinMax for clarity.
            
            # Calls MIN node
            val = self.min_value(board, opponent, depth - 1, alpha, beta, player)
            
            board.board[mx][my] = 0 # Undo
            
            if val > best_val:
                best_val = val
                best_move = move
            
            # Pruning
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break 

        return best_val, best_move

    def min_value(self, board, player, depth, alpha, beta, original_player):
        # Check timeout
        if (self.nodes_explored & 1023) == 0:
             if time.time() - self.start_time > self.time_limit: raise TimeoutError()
        self.nodes_explored += 1

        result = board.get_game_result()
        if result != 0:
            if result == original_player: return SCORE_FIVE * 10
            if result == (3 - original_player): return -SCORE_FIVE * 10
            return 0
            
        if depth == 0:
            return self.evaluate_board(board, original_player)

        opponent = 3 - player
        candidates = self.get_sorted_moves(board, player)
        if not candidates: return 0

        v = math.inf
        for mx, my in candidates:
            board.place_stone(mx, my, player)
            # Back to Max
            val, _ = self.alpha_beta_search(board, opponent, depth - 1, alpha, beta)
            # wait, alpha_beta_search IS the max function wrapper. 
            # Actually simpler to just write max_value separate or use negamax.
            # Let's use logic:
            # We are in MIN node. We want to minimize 'original_player's score.
            # Current player is 'player' (the opponent of original_player).
            
            # BUT my alpha_beta_search returns (BestScoreForPlayer, BestMove).
            # If I call it for 'opponent' (which is 'original_player'), it returns Score for OriginalPlayer (MAX).
            # This works.
            
            # val is Score from OriginalPlayer perspective
            val = val # It already returns max score for that player
            
            # But wait, alpha_beta_search computes score for "current player passed to it".
            # If we pass 'opponent' (who is original_player), it maximizes for original_player.
            # That's exactly what we want? No, we are MIN node. We want to choose move that MINIMIZES original_player score.
            # So if alpha_beta_search returns HIGH score for original_player, we avoid it?
             
            # Let's fix recursion logic properly:
            # AlphaBeta(p) -> returns max score for p.
            # Here we are opponent. We want to MAXIMIZE opponent's score.
            # Score for Opponent = - Score for OriginalPlayer.
            
            # Simpler implementation: NegaMax
            # Let's stick to simple "evaluate always from Player 1 perspective".
            # Assume 'original_player' is the Hero.
            # If current player == Hero: Maximize Eval.
            # If current player != Hero: Minimize Eval.
            pass # Logic fixed below in cleaner structure
            
            board.board[mx][my] = 0
            v = min(v, val)
            if v <= alpha: return v
            beta = min(beta, v)
        return v
    
    # REWRITE FOR CLEAN NEGAMAX to avoid confusion
    # Evaluation is always from 'player' perspective.
    # NegaMax: max( - child_val )
    
    def negamax(self, board, depth, alpha, beta, color) -> float:
        # Check timeout
        if (self.nodes_explored & 1023) == 0:
             if time.time() - self.start_time > self.time_limit: raise TimeoutError()
        self.nodes_explored += 1
        
        # Winner check
        res = board.get_game_result()
        if res != 0:
            if res == color: return SCORE_FIVE * (1 + depth) # Prefer winning sooner
            if res == (3-color): return -SCORE_FIVE * (1 + depth)
            return 0
            
        if depth == 0:
            return self.evaluate_shape(board, color)
            
        candidates = self.get_sorted_moves(board, color)
        if not candidates: return 0
        
        value = -math.inf
        for mx, my in candidates:
            try:
                board.place_stone(mx, my, color)
                val = -self.negamax(board, depth - 1, -beta, -alpha, 3 - color)
                value = max(value, val)
                alpha = max(alpha, value)
            finally:
                board.board[mx][my] = 0
                board.move_count -= 1

            if alpha >= beta:
                break
        return value

    def alpha_beta_search(self, board, player, depth, alpha, beta):
        # Root caller for NegaMax to retrieve the move
        candidates = self.get_sorted_moves(board, player)
        if not candidates: return 0, (-1, -1)
        
        best_val = -math.inf
        best_move = candidates[0]
        
        for mx, my in candidates:
            # Pre-check immediate win to save time
            board.place_stone(mx, my, player)
            if board.get_game_result() == player:
                board.board[mx][my] = 0
                board.move_count -= 1
                return SCORE_FIVE, (mx, my)
            
            # Search
            val = -self.negamax(board, depth - 1, -beta, -alpha, 3 - player)
            board.board[mx][my] = 0
            board.move_count -= 1
            
            if val > best_val:
                best_val = val
                best_move = (mx, my)
            alpha = max(alpha, best_val)
        
        return best_val, best_move

    def get_sorted_moves(self, board: Board, player: int) -> List[Tuple[int, int]]:
        # Get neighbors distance 2
        moves = set()
        size = board.size
        # Fast neighbor implementation or rely on existing (but I want this file self-contained for speed)
        # Scan used cells locally
        for x in range(size):
            for y in range(size):
                if board.board[x][y] != 0:
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x+dx, y+dy
                            if board.is_valid_move(nx, ny):
                                moves.add((nx, ny))
        
        candidates = list(moves)
        if not candidates: return [(size//2, size//2)]
        
        # Heuristic Sort: Prioritize moves that form shape or block shape
        # We use a simplified shape scorer for sorting
        scored = []
        for mx, my in candidates:
            # Bias towards center
            bias = 7 - max(abs(mx - 7), abs(my - 7))
            score = bias
            # Check immediate adjacent
            # (Too expensive to do full eval here, use simple proximity)
            scored.append((score, (mx, my)))
            
        # Sort descending
        # Actually random shuffle + center bias is okay for shallow
        # But let's check for kill moves (Winning moves)
        
        scored.sort(key=lambda x: x[0], reverse=True)
        # Limit branching factor for speed? 
        # For Top Level we want all. For deeper, we limit.
        return [x[1] for x in scored[:20]] # Only check top 20 relevant moves

    def evaluate_shape(self, board: Board, player: int) -> float:
        score = 0
        opponent = 3 - player
        
        # Scan all lines
        # This is the heavy part. Optimized version scans board once.
        # Here we use a slightly simpler version:
        
        # Horizontal
        for x in range(board.size):
            row = board.board[x]
            score += self.evaluate_line(row, player, opponent)
            
        # Vertical
        for y in range(board.size):
            col = [board.board[x][y] for x in range(board.size)]
            score += self.evaluate_line(col, player, opponent)
            
        # Diagonals (Top-Left to Bottom-Right)
        # We collect all diagonals
        # Diagonals are identified by (x - y) = k. k ranges from -(size-1) to (size-1)
        for k in range(-(board.size - 1), board.size):
            line = []
            for x in range(board.size):
                y = x - k
                if 0 <= y < board.size:
                    line.append(board.board[x][y])
            if len(line) >= 5:
                score += self.evaluate_line(line, player, opponent)

        # Anti-Diagonals (Top-Right to Bottom-Left)
        # Identified by (x + y) = k. k ranges from 0 to 2*(size-1)
        for k in range(2 * board.size - 1):
            line = []
            for x in range(board.size):
                y = k - x
                if 0 <= y < board.size:
                    line.append(board.board[x][y])
            if len(line) >= 5:
                score += self.evaluate_line(line, player, opponent)
        
        return score

    def evaluate_line(self, line: List[int], player: int, opponent: int) -> int:
        score = 0
        s = "".join(str(x) for x in line)
        
        # Patterns for Player
        p = str(player)
        score += s.count(p*5) * SCORE_FIVE
        score += s.count("0"+p*4+"0") * SCORE_LIVE_4
        score += s.count("0"+p*4+str(opponent)) * SCORE_DEAD_4
        score += s.count(str(opponent)+p*4+"0") * SCORE_DEAD_4
        score += s.count("0"+p*3+"0") * SCORE_LIVE_3
        
        # Patterns for Opponent (Subtract Score)
        o = str(opponent)
        score -= s.count(o*5) * SCORE_FIVE * 1.2 # Fear defeat more than victory
        score -= s.count("0"+o*4+"0") * SCORE_LIVE_4 * 1.2
        score -= s.count("0"+o*3+"0") * SCORE_LIVE_3 * 1.5 # Block live 3s!
        
        return score
