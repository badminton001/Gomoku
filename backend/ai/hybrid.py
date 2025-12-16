"""
Hybrid AI Agent (Policy-Guided AlphaBeta).

Combines:
1.  Policy Network (SL Model) for intuitive candidate suggestions.
2.  Alpha-Beta Search (Depth 2) for tactical verification.
3.  Heuristic Defense blocks for immediate threats.
"""
import torch
import numpy as np
import time
import math
from typing import Tuple, List
from backend.engine.board import Board
from backend.ai.minimax import AlphaBetaAgent, SCORE_FIVE, SCORE_LIVE_4, SCORE_LIVE_3
from backend.ai.policy_network import Net

class HybridAgent:
    """
    Hybrid Agent combining Policy Network with tactical search.
    """
    def __init__(self, model_path: str = "models/sl_policy_v1_base.pth", device="cpu"):
        self.device = device
        self.model = Net().to(device)
        self.model.eval()
        try:
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"[HybridAgent] Loaded {model_path}")
        except Exception as e:
            print(f"[HybridAgent] Warning: Failed to load model {model_path}: {e}")
            self.model = None

        # Search Engine for Verification
        self.search_engine = AlphaBetaAgent(depth=2, time_limit=1.0)
        self.logs = [] # Debug Logs

    def save_logs(self, filename):
        with open(filename, "w") as f:
            for line in self.logs:
                f.write(line + "\n")
        print(f"[HybridAgent] Logs saved to {filename}")

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        log_entry = f"--- Turn {board.move_count // 2 + 1} (Player {player}) ---"
        self.logs.append(log_entry)
        
        # 1. Immediate Win/Block Check
        win_kill = self._check_immediate_win_loss(board, player)
        if win_kill: 
            self.logs.append(f"Immediate Win/Block found: {win_kill}")
            return win_kill

        # 2. Critical Defense (Block Live 4s/3s)
        opp = 3 - player
        critical_blocks = self._get_critical_defenses(board, opp)
        
        if critical_blocks:
             self.logs.append(f"  [Defense] Forced Defense: {list(critical_blocks)}")
             candidates = list(critical_blocks)
        else:
            # 3. Generate Candidates (SL + Heuristic)
            candidates = self._get_candidates(board, player)
        
        # 3b. Soft Defense (Threat Reduction)
        if not critical_blocks:
            current_threat = self.search_engine.evaluate_shape(board, opp)
            if current_threat >= SCORE_LIVE_3 * 0.8:
                candidates = self._filter_defensive_moves(board, candidates, player, current_threat)
            
        if not candidates:
            # Fallback
            for x in range(15):
                for y in range(15):
                    if board.is_valid_move(x, y): return x, y
            return (-1, -1)

        # 4. Deep Verification (AlphaBeta)
        best_move = self._verify_candidates(board, candidates, player)
        self.logs.append(f"Selected Move: {best_move}")
        
        # 5. Final Sanity Check
        if not board.is_valid_move(best_move[0], best_move[1]):
            self.logs.append(f"[Panic] Invalid move {best_move}. Random search.")
            for x in range(15):
                for y in range(15):
                    if board.is_valid_move(x, y): return x, y
            return (-1, -1)
            
        return best_move

    def get_top_moves(self, board, player, limit=5):
        """Returns top candidates with policy confidence."""
        if not self.model: 
            return []
            
        inp = np.zeros((15, 15), dtype=np.float32)
        b_np = np.array(board.board)
        inp[b_np == player] = 1.0
        inp[(b_np != 0) & (b_np != player)] = -1.0
        
        t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(self.device)
        candidates = []
        with torch.no_grad():
            logits = self.model(t_inp)
            probs = torch.softmax(logits, dim=1)
            
            top_probs, top_indices = torch.topk(probs, limit, dim=1)
            flat_probs = top_probs[0]
            flat_indices = top_indices[0]
            
            for i in range(len(flat_probs)):
                score = flat_probs[i].item()
                idx = flat_indices[i].item()
                x, y = divmod(idx, 15)
                if board.is_valid_move(x, y):
                    candidates.append((score, (x, y)))
        return candidates

    def _verify_candidates(self, board, candidates, player):
        best_val = -math.inf
        best_move = candidates[0]
        alpha = -math.inf
        beta = math.inf
        
        self.search_engine.start_time = time.time()
        log_scores = []
        
        for mx, my in candidates:
            if not board.place_stone(mx, my, player): continue
            
            # Negamax Search (Depth 2)
            try:
                val = -self.search_engine.negamax(board, depth=2, alpha=-beta, beta=-alpha, color=3-player)
            except Exception:
                val = -math.inf
                
            board.board[mx][my] = 0
            board.move_count -= 1
            
            log_scores.append(f"Move ({mx},{my}): {val}")
            
            if val > best_val:
                best_val = val
                best_move = (mx, my)
            alpha = max(alpha, best_val)
        
        self.logs.append(f"Verification Scores (Top 5): {log_scores[:5]} ...")
        return best_move

    def _get_critical_defenses(self, board, opponent):
        """Scan for opponent's threats (Live 3/4) that must be blocked."""
        blocks = set()
        size = board.size
        
        for x in range(size):
            for y in range(size):
                if board.board[x][y] == 0:
                     # Simulate opponent move
                     board.board[x][y] = opponent
                     if self._is_live_four(board, x, y, opponent):
                         blocks.add((x, y))
                     board.board[x][y] = 0
        return blocks

    def _is_live_four(self, board, x, y, color):
        """Check if move creates a live four (011110) pattern."""
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        size = 15
        p_str = str(color)
        pattern_live = "0" + p_str*4 + "0"
        
        for dx, dy in dirs:
            line_str = ""
            for k in range(-4, 5):
                nx, ny = x + k*dx, y + k*dy
                if 0 <= nx < size and 0 <= ny < size:
                    line_str += str(board.board[nx][ny])
                else:
                    line_str += "X"
            
            if pattern_live in line_str:
                return True
        return False

    def _get_candidates(self, board: Board, player: int) -> List[Tuple[int, int]]:
        candidates = []
        seen = set()

        # A. Neural Network Proposals
        if self.model:
            inp = np.zeros((15, 15), dtype=np.float32)
            b_np = np.array(board.board)
            inp[b_np == player] = 1.0
            inp[(b_np != 0) & (b_np != player)] = -1.0
            
            t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(t_inp)
                probs = torch.softmax(logits, dim=1)
                
                # Top-K Sampling
                top_k = 20
                top_probs, top_indices = torch.topk(probs, top_k, dim=1)
                flat_probs = top_probs[0] / torch.sum(top_probs[0])
                flat_indices = top_indices[0]
                
                # Sample 8 candidates
                sample_indices = torch.multinomial(flat_probs, 8, replacement=False)
                
                for s_idx in sample_indices:
                     real_idx = flat_indices[s_idx].item()
                     x, y = divmod(real_idx, 15)
                     if board.is_valid_move(x, y):
                         candidates.append((x, y))
                         seen.add((x, y))

        # B. Heuristic Proposals
        heuristic_moves = self.search_engine.get_sorted_moves(board, player)
        count = 0
        for m in heuristic_moves:
            if m not in seen:
                candidates.append(m)
                seen.add(m)
                count += 1
                if count >= 15: break
                
        return candidates

    def _check_immediate_win_loss(self, board: Board, player: int) -> Tuple[int, int]:
        # Check Win
        for x in range(15):
            for y in range(15):
                if board.board[x][y] == 0:
                    if self._is_seq_5(board, x, y, player): return (x, y)
        
        # Check Loss
        opp = 3 - player
        for x in range(15):
            for y in range(15):
                if board.board[x][y] == 0:
                     if self._is_seq_5(board, x, y, opp): return (x, y)
        return None

    def _is_seq_5(self, board, x, y, color):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in dirs:
            c = 1
            # Forward
            nx, ny = x+dx, y+dy
            while 0<=nx<15 and 0<=ny<15 and board.board[nx][ny] == color: c+=1; nx+=dx; ny+=dy
            # Backward
            nx, ny = x-dx, y-dy
            while 0<=nx<15 and 0<=ny<15 and board.board[nx][ny] == color: c+=1; nx-=dx; ny-=dy
            if c >= 5: return True
        return False

    def _filter_defensive_moves(self, board, candidates, player, current_threat):
        opp = 3 - player
        defensive = []
        for mx, my in candidates:
            board.board[mx][my] = player
            new_threat = self.search_engine.evaluate_shape(board, opp)
            board.board[mx][my] = 0
            
            if new_threat < current_threat:
                defensive.append((new_threat, (mx, my)))
                
        defensive.sort(key=lambda x: x[0])
        best_defensive = [x[1] for x in defensive[:4]]
        return best_defensive if best_defensive else candidates
