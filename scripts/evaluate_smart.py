import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time
import math

sys.path.append(os.getcwd())
from backend.engine.game_engine import GameEngine
from backend.ai.basic.classic_ai import RandomAgent, GreedyAgent
from backend.ai.basic.strong_ai import AlphaBetaAgent, SCORE_FIVE, SCORE_LIVE_4, SCORE_DEAD_4, SCORE_LIVE_3

# --- DEFINE NETWORK ---
class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 2, 1), nn.BatchNorm2d(2), nn.ReLU()
        )
        self.fc = nn.Linear(2*15*15, 15*15)
        
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class DeepSmartAgent:
    """
    Policy-Guided AlphaBeta Search (AlphaZero Lite):
    1. Neural Network proposes Top K strategic moves.
    2. Deep AlphaBeta (Depth 4) verifies them.
    """
    def __init__(self, model_path, device="cpu"):
        self.model = Net().to(device)
        self.device = device
        self.model.eval()
        try:
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"[DeepSmart] Loaded {model_path}")
        except Exception as e:
            print(f"[DeepSmart] Error: {e}")
            
        # Use Strong AI for Search Logic
        self.bg_search = AlphaBetaAgent(depth=2) # Helper

    def get_move(self, board, player):
        # 1. Immediate Win/Block (The "Oh no you don't" Check)
        for x in range(board.size):
            for y in range(board.size):
                if board.is_valid_move(x, y):
                    # Win?
                    board.place_stone(x, y, player)
                    if board.check_five_in_a_row(player):
                        board.board[x][y] = 0
                        board.move_count -= 1
                        return x, y
                    board.board[x][y] = 0
                    board.move_count -= 1
                    # Block?
                    opp = 3 - player
                    board.place_stone(x, y, opp)
                    if board.check_five_in_a_row(opp):
                        board.board[x][y] = 0
                        board.move_count -= 1
                        return x, y
                    board.board[x][y] = 0
                    board.move_count -= 1

        # 2. Candidate Generation (NN + Tactical Sentinel)
        
        # A. Neural Network (Intuition)
        inp = np.zeros((15, 15), dtype=np.float32)
        for r in range(15):
            for c in range(15):
                v = board.board[r][c]
                if v == player: inp[r][c] = 1.0
                elif v != 0: inp[r][c] = -1.0
        
        t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(self.device)
        nn_candidates = []
        with torch.no_grad():
            logits = self.model(t_inp)
            probs = torch.softmax(logits, dim=1)
            sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]
            count = 0
            for idx in sorted_indices:
                idx = idx.item()
                x, y = divmod(idx, 15)
                if board.is_valid_move(x, y):
                    nn_candidates.append((x, y))
                    count += 1
                if count >= 8: break 
        
        # B. Tactical Sentinel (Heuristic Calculation)
        # Scan neighbors and score them like GreedyAgent (Attack + Defense)
        # This ensures we don't miss obvious tactical moves that NN might miss.
        from backend.ai.advanced.mcts_ai import get_neighbor_moves
        neighbors = get_neighbor_moves(board, distance=2)
        tactical_scores = []
        
        # Simple directional evaluator (Inline for speed)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        opp = 3 - player
        
        def fast_score(bx, by, target):
            s = 0
            board.board[bx][by] = target
            for dx, dy in directions:
                count = 1
                # Forward
                tx, ty = bx + dx, by + dy
                while board.is_inside(tx, ty) and board.board[tx][ty] == target:
                    count += 1; tx += dx; ty += dy
                # Backward
                tx, ty = bx - dx, by - dy
                while board.is_inside(tx, ty) and board.board[tx][ty] == target:
                    count += 1; tx -= dx; ty -= dy
                
                if count >= 5: s += 100000
                elif count == 4: s += 10000
                elif count == 3: s += 1000
                elif count == 2: s += 100
            board.board[bx][by] = 0
            return s

        for mx, my in neighbors:
            if board.is_valid_move(mx, my):
                atk = fast_score(mx, my, player)
                dfs = fast_score(mx, my, opp)
                score = atk + dfs
                tactical_scores.append((score, (mx, my)))
        
        tactical_scores.sort(key=lambda x: x[0], reverse=True)
        heuristic_candidates = [x[1] for x in tactical_scores[:5]] # Top 5 Tactical

        # C. Merge Candidates
        candidates = []
        seen = set()
        for m in nn_candidates:
            if m not in seen: candidates.append(m); seen.add(m)
        for m in heuristic_candidates:
            if m not in seen: candidates.append(m); seen.add(m)
            
        if not candidates: return (7, 7)

        # 3. Threat Pruning (Focus Mode)
        current_threat = self.bg_search.evaluate_shape(board, opp)
        if current_threat >= SCORE_LIVE_3 * 0.9: 
            print(f"  [Defense] Threat Detected ({current_threat})...")
            defensive_moves = []
            for mx, my in candidates:
                board.place_stone(mx, my, player)
                new_threat = self.bg_search.evaluate_shape(board, opp)
                board.board[mx][my] = 0
                board.move_count -= 1
                if new_threat < current_threat:
                     defensive_moves.append((new_threat, (mx, my)))
            
            defensive_moves.sort(key=lambda x: x[0])
            candidates = [x[1] for x in defensive_moves[:4]]
            if not candidates:
                 # Panic: Search ALL neighbors if our limited pool has no defense
                 print("  [Panic] Standard candidates failed. Checking ALL neighbors...")
                 all_defensive = []
                 for mx, my in neighbors:
                     if board.is_valid_move(mx, my):
                        board.place_stone(mx, my, player)
                        nt = self.bg_search.evaluate_shape(board, opp)
                        board.board[mx][my] = 0
                        board.move_count -= 1
                        if nt < current_threat:
                            all_defensive.append((nt, (mx, my)))
                 all_defensive.sort(key=lambda x: x[0])
                 candidates = [x[1] for x in all_defensive[:3]]
        
        if not candidates: candidates = [heuristic_candidates[0]] if heuristic_candidates else [(7,7)]

        # 4. Deep Verification
        # print(f"Thinking... Searching {len(candidates)} moves...") # (Reduced spam)
        
        best_val = -math.inf
        best_move = candidates[0]
        
        alpha = -math.inf
        beta = math.inf
        
        import copy
        search_board = copy.deepcopy(board) # ISOLATE SEARCH
        
        for mx, my in candidates:
            search_board.place_stone(mx, my, player)
            
            self.bg_search.start_time = time.time() + 100 
            self.bg_search.nodes_explored = 0
            
            try:
                # Search Depth 2 
                val = -self.bg_search.negamax(search_board, depth=2, alpha=-beta, beta=-alpha, color=3-player)
            except Exception as e:
                print(f"Search Error: {e}")
                val = -math.inf
            
            search_board.board[mx][my] = 0
            search_board.move_count -= 1
            
            if val > best_val:
                best_val = val
                best_move = (mx, my)
            
            alpha = max(alpha, best_val)
        
        return best_move

def print_board_result(engine):
    b = engine.board.board
    size = engine.board.size
    print("\n   " + " ".join([f"{i:X}" for i in range(size)]))
    for r in range(size):
        row_str = f"{r:X}  "
        for c in range(size):
            v = b[r][c]
            if v == 0: row_str += ". "
            elif v == 1: row_str += "X " # P1
            elif v == 2: row_str += "O " # P2
        print(row_str)
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs", choices=["greedy", "strong"], default="greedy")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_smart = DeepSmartAgent("models/sl_model_v1.pth", device=device)
    
    if args.vs == "greedy":
        opponent = GreedyAgent()
    elif args.vs == "strong":
        opponent = AlphaBetaAgent(depth=2, time_limit=1.0) # Baseline Strong
        
    print(f"\n>>> DEEP SMART SL vs {args.vs.upper()} <<<")
    
    engine = GameEngine()
    engine.reset_game()
    
    # Smart Agent is P1 (Black)
    start_t = time.time()
    while not engine.game_over:
        p = engine.current_player
        if p == 1:
            x, y = agent_smart.get_move(engine.board, p)
        else:
            x, y = opponent.get_move(engine.board, p)
            
        if not engine.make_move(x, y):
             print(f"Invalid move by {p}")
             break
        if engine.board.move_count % 5 == 0:
            print(f"Move {engine.board.move_count}...")
            
    print(f"Game Over in {time.time()-start_t:.1f}s. Winner: {engine.winner}")
    print_board_result(engine) # Visualize
    
    if engine.winner == 1:
        print("VERDICT: VICTORY!")
    elif engine.winner == 2:
        print("VERDICT: DEFEAT")
    else:
        print("VERDICT: DRAW")

if __name__ == "__main__":
    main()
