import torch
import numpy as np
import time
import math
from typing import Tuple, List
from backend.engine.board import Board
from backend.ai.basic.strong_ai import AlphaBetaAgent, SCORE_FIVE, SCORE_LIVE_4, SCORE_LIVE_3
from backend.ai.basic.sl_network import Net

class HybridAgent:
    """
    Policy-Guided AlphaBeta Search (Hybrid AI):
    - Policy Network (SL Model) provides "Intuitive Candidates"
    - Heuristics provide "Tactical Candidates"
    - Shallow AlphaBeta (Depth 2) verifies and selects the best move.
    """
    def __init__(self, model_path: str = "models/sl_model_v1.pth", device="cpu"):
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
        
        # 1. Immediate Win/Block Check (Fastest)
        win_kill = self._check_immediate_win_loss(board, player)
        if win_kill: 
            self.logs.append(f"Immediate Win/Block found: {win_kill}")
            return win_kill

        # 2. Hard Defense (Block Live 3s)
        # Scan for opponent threats that MUST be answered
        # If we ignore a Live 3, they get Live 4 -> Win
        opp = 3 - player
        critical_blocks = self._get_critical_defenses(board, opp)
        
        # If critical blocks exist, we RESTRICT candidates to these blocks
        # We can still check which block is best using Search
        if critical_blocks:
             msg = f"  [Defense] Forced Defense against {len(critical_blocks)} threats: {list(critical_blocks)}"
             # print(msg) # Squelch console noise, logs have this
             self.logs.append(msg)
             candidates = list(critical_blocks)
        else:
            # 3. Generate SL/Heuristic Candidates (Normal Play)
            candidates = self._get_candidates(board, player)
        
        # 3b. Threat Analysis (Soft Defense for minor threats)
        # If no critical immediate death, we still check shape score
        if not critical_blocks:
            current_threat = self.search_engine.evaluate_shape(board, opp)
            if current_threat >= SCORE_LIVE_3 * 0.8:
                candidates = self._filter_defensive_moves(board, candidates, player, current_threat)
            
        # If no candidates left (shouldn't happen if blocks exist), fallback
        if not candidates:
            # Fallback: Find ANY valid move to avoid crash
            for x in range(15):
                for y in range(15):
                    if board.is_valid_move(x, y):
                        return x, y
            return (-1, -1) # Board full

        # 4. Deep Verification (AlphaBeta Search on Candidates)
        best_move = self._verify_candidates(board, candidates, player)
        
        self.logs.append(f"Selected Move: {best_move}")
        
        # 5. Final Sanity Check
        # Ensure we never return an invalid move (e.g. from board corruption or logic error)
        bx, by = best_move
        if not board.is_valid_move(bx, by):
            err = f"[HybridAgent] Panic: Proposed invalid move {best_move}. Finding random fallback."
            print(err)
            self.logs.append(err)
            for x in range(15):
                for y in range(15):
                    if board.is_valid_move(x, y):
                        return x, y
            return (-1, -1)
            
        return best_move

    def _verify_candidates(self, board, candidates, player):
        best_val = -math.inf
        best_move = candidates[0]
        
        alpha = -math.inf
        beta = math.inf
        
        # FIX: Must initialize start_time, otherwise negamax thinks time is up (Epoch 0)
        self.search_engine.start_time = time.time()
        
        log_scores = []
        
        for mx, my in candidates:
            if not board.place_stone(mx, my, player):
                 continue
            
            # Negamax Search
            # Depth 2 + Paranoid Eval is enough to see threats 1 move ahead
            # Depth 4 is too slow for Python
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
        # Scan for Opponent's "Live 3" or "Split 3" or "Dead 4"
        # These are threats that become "Win in 1" or "Win in 2 (Unstoppable)" if not handled.
        # Returns a Set of valid blocking coordinates.
        
        blocks = set()
        size = board.size
        # Directions
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        
        for x in range(size):
            for y in range(size):
                if board.board[x][y] == 0:
                     # Hypothetical: Opponent moves here
                     board.board[x][y] = opponent
                     
                     if self._is_live_four(board, x, y, opponent):
                         blocks.add((x, y))
                         
                     board.board[x][y] = 0
        return blocks



    def _is_live_four(self, board, x, y, color):
        # Check if (x,y) participates in a Live 4 or Split 4
        # We need to detect:
        # 0XXXX0 (Standard Live 4)
        # 0X.XXX0 (Split 4) - actually this is a "Dead 4" technically if blocked? 
        # No, X.XXX is 4 stones. If I place stone at ., I get XXXXX (5).
        # Wait, the logic here is: I am Hypothesizing Opponent Move at (x,y).
        # So board has Opponent at (x,y).
        # I want to know if this creates a "Live 4" (011110) which implies "Unstoppable 5".
        # Or if it creates "5" (11111).
        
        # Actually, "Critical Defense" means:
        # 1. Opponent has 4 -> Will be 5. (Immediate Loss) -> blocked by _check_immediate_win_loss ?
        #    - _check_immediate_win_loss ONLY checks if opponent HAS 4 on board (threatens 5).
        #    - What if opponent has 3 (Live 3) and plays to make it Live 4?
        #    - That is what we catch here.
        # So we look for: Resulting pattern contains "011110".
        
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        size = 15
        p_str = str(color)
        pattern_live = "0" + p_str*4 + "0"
        
        for dx, dy in dirs:
            # Extract line of length 9 centered at x,y (4 radius)
            # This is enough to see 0XXXX0
            line_str = ""
            # We need to range from -4 to +4
            for k in range(-4, 5):
                nx, ny = x + k*dx, y + k*dy
                if 0 <= nx < size and 0 <= ny < size:
                    v = board.board[nx][ny]
                    line_str += str(v)
                else:
                    line_str += "X" # Boundary
            
            if pattern_live in line_str:
                return True
                
        return False

    def _get_candidates(self, board: Board, player: int) -> List[Tuple[int, int]]:
        candidates = []
        seen = set()


        # A. Neural Network Proposals (Top 10 with Softmax Sampling)
        if self.model:
            inp = np.zeros((15, 15), dtype=np.float32)
            # Vectorized board encoding
            b_np = np.array(board.board)
            inp[b_np == player] = 1.0
            inp[(b_np != 0) & (b_np != player)] = -1.0
            
            t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(t_inp)
                probs = torch.softmax(logits, dim=1)
                
                # --- Stochastic Sampling ---
                # Get Top 20 to select from
                top_k = 20
                top_probs, top_indices = torch.topk(probs, top_k, dim=1)
                
                # Flatten batch dim
                flat_probs = top_probs[0] # (20,)
                flat_indices = top_indices[0] # (20,)
                
                # Re-normalize
                flat_probs = flat_probs / torch.sum(flat_probs)
                
                # Sample 8 candidates
                # returns 1D tensor of indices (8,)
                sample_indices = torch.multinomial(flat_probs, 8, replacement=False)
                
                for s_idx in sample_indices:
                     # s_idx is index into the Top 20 array
                     real_idx = flat_indices[s_idx].item()
                     x, y = divmod(real_idx, 15)
                     if board.is_valid_move(x, y):
                         candidates.append((x, y))
                         seen.add((x, y))

        # B. Tactical Sentinel (Heuristic Proposals)
        # Add moves that create 3s/4s or block them
        # We reuse AlphaBetaAgent's sorting logic which is efficient
        heuristic_moves = self.search_engine.get_sorted_moves(board, player)
        count = 0
        for m in heuristic_moves:
            if m not in seen:
                candidates.append(m)
                seen.add(m)
                count += 1
                if count >= 15: break # Reduced to 15 for speed
                
        return candidates


    def _check_immediate_win_loss(self, board: Board, player: int) -> Tuple[int, int]:
        # Reuse the fast check logic if available or scan empty cells
        # We can implement a fast scanner here.
        
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
        # Quick helper
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in dirs:
            c = 1
            # Fwd
            nx, ny = x+dx, y+dy
            while 0<=nx<15 and 0<=ny<15 and board.board[nx][ny] == color: c+=1; nx+=dx; ny+=dy
            # Bwd
            nx, ny = x-dx, y-dy
            while 0<=nx<15 and 0<=ny<15 and board.board[nx][ny] == color: c+=1; nx-=dx; ny-=dy
            if c >= 5: return True
        return False

    def _filter_defensive_moves(self, board, candidates, player, current_threat):
        opp = 3 - player
        defensive = []
        for mx, my in candidates:
            # Try move
            board.board[mx][my] = player # Manual set for speed
            # Check if threat reduced
            new_threat = self.search_engine.evaluate_shape(board, opp)
            board.board[mx][my] = 0
            
            if new_threat < current_threat:
                defensive.append((new_threat, (mx, my)))
                
        defensive.sort(key=lambda x: x[0])
        best_defensive = [x[1] for x in defensive[:4]]
        
        if not best_defensive:
             # Panic Mode: Check all neighbors if candidates failed
             # (Simplified for now: just return candidates and hope AlphaBeta finds the block)
             pass
             
        return best_defensive if best_defensive else candidates


