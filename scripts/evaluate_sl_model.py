import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.engine.game_engine import GameEngine
from backend.ai.basic.classic_ai import RandomAgent, GreedyAgent
try:
    from backend.ai.basic.strong_ai import AlphaBetaAgent
except ImportError:
    AlphaBetaAgent = None

def is_winning_move(board, x, y, player):
    """Check if placing a stone at (x,y) for 'player' results in a win (5 in a row).
    This is much faster than scanning the whole board."""
    # Temporarily place stone
    # Note: We assume board.board[x][y] is 0 before calling or logic handles it?
    # No, we simulate.
    
    # Directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        # Forward
        cx, cy = x + dx, y + dy
        while 0 <= cx < board.size and 0 <= cy < board.size and board.board[cx][cy] == player:
            count += 1
            cx += dx
            cy += dy
        # Backward
        cx, cy = x - dx, y - dy
        while 0 <= cx < board.size and 0 <= cy < board.size and board.board[cx][cy] == player:
            count += 1
            cx -= dx
            cy -= dy
        
        if count >= 5:
            return True
    return False

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
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

class SLAgent:
    def __init__(self, model_path, device="cpu"):
        self.model = Net().to(device)
        self.device = device
        self.model.eval()
        try:
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"[SLAgent] Loaded {model_path}")
        except Exception as e:
            print(f"[SLAgent] Failed to load model: {e}")

    def get_move(self, board, player):
        # --- 1. Blunder Guard (Logic Layer) ---
        # Pure SL sometimes misses "obvious" 4-in-a-row threats. 
        # We add a simple check: "Can I win now?" or "Must I block now?"
        
        # Check for immediate win (My 4 -> 5)
        # Check for immediate win (My 4 -> 5)
        for x in range(board.size):
            for y in range(board.size):
                if board.board[x][y] == 0: # Is empty
                     if is_winning_move(board, x, y, player):
                         return x, y
        
        # Check for immediate loss (Opponent 4 -> 5)
        opponent = 3 - player
        for x in range(board.size):
            for y in range(board.size):
                if board.board[x][y] == 0:
                     if is_winning_move(board, x, y, opponent):
                         return x, y # Must block!
        
        # --- 2. Neural Network Intuition ---
        inp = np.zeros((15, 15), dtype=np.float32)
        for r in range(15):
            for c in range(15):
                v = board.board[r][c]
                if v == player: inp[r][c] = 1.0
                elif v != 0: inp[r][c] = -1.0
        
        t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t_inp)
            probs = torch.softmax(logits, dim=1)
            
            # --- Stochastic Sampling (Variety) ---
            # Instead of always taking the best, we sample from the distribution
            # This allows us to see different games
            
            # Get Top 10 candidates to avoid picking garbage
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            
            # Re-normalize to sum to 1
            top_probs = top_probs / torch.sum(top_probs)
            
            # Sample
            sample_idx = torch.multinomial(top_probs, 1).item()
            best_idx = top_indices[0][sample_idx].item()
            
            # Try the sampled move first
            x, y = divmod(best_idx, 15)
            if board.is_valid_move(x, y):
                return x, y
                
            # Fallback: Just iterate sorted if the sampled one was invalid (rare)
            sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]
            for idx in sorted_indices:
                idx = idx.item()
                x, y = divmod(idx, 15)
                if board.is_valid_move(x, y):
                    return x, y
        return -1, -1

class HumanAgent:
    def get_move(self, board, player):
        while True:
            try:
                s = input(f"Your Move (Player {player}) [row col]: ")
                parts = s.split()
                if len(parts) != 2: continue
                x, y = int(parts[0]), int(parts[1])
                if board.is_valid_move(x, y): return x, y
                print("Invalid move.")
            except ValueError:
                pass

def print_board(board):
    print("   " + " ".join([f"{i:2}" for i in range(15)]))
    for r in range(15):
        row = []
        for c in range(15):
            v = board.board[r][c]
            if v == 0: row.append(".")
            elif v == 1: row.append("X") # SL Agent is Black (usually)
            else: row.append("O")
        print(f"{r:2} " + "  ".join(row))

def play_game(agent1, agent2, render=False, delay=0.0):
    engine = GameEngine()
    engine.reset_game()
    
    while not engine.game_over:
        if render:
            print_board(engine.board)
            print("-" * 30)
            time.sleep(delay)
            
        p = engine.current_player
        if p == 1:
            x, y = agent1.get_move(engine.board, p)
        else:
            x, y = agent2.get_move(engine.board, p)
            
        if not engine.make_move(x, y):
            print(f"Player {p} made Invalid Move ({x},{y})! Game Over.")
            return 3 - p, engine.board.move_count # Opponent wins
            
    if render:
        print_board(engine.board)
        print(f"Winner: {engine.winner}")
        
    return engine.winner, engine.board.move_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="Play against the AI")
    parser.add_argument("--vs", choices=["random", "greedy", "strong", "all"], default="strong", help="Opponent type")
    parser.add_argument("--games", type=int, default=1, help="Number of games to simulate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_sl = SLAgent("models/sl_model_v1.pth", device=device)
    
    if args.human:
        # ... human code ...
        print("\n>>> HUMAN vs SL AI CHALLENGE <<<")
        print("You are Player 2 (O). AI is Player 1 (X).")
        opponent = HumanAgent()
        play_game(agent_sl, opponent, render=True)
        return

    # Define opponents to run
    opponents_map = {}
    if args.vs == "all":
        opponents_map["Random"] = RandomAgent()
        opponents_map["Greedy"] = GreedyAgent()
        if AlphaBetaAgent:
            opponents_map["Strong"] = AlphaBetaAgent(depth=2, time_limit=1.0)
    elif args.vs == "random":
        opponents_map["Random"] = RandomAgent()
    elif args.vs == "greedy":
        opponents_map["Greedy"] = GreedyAgent()
    elif args.vs == "strong":
        if AlphaBetaAgent:
             opponents_map["Strong"] = AlphaBetaAgent(depth=2, time_limit=1.0)
        else:
             print("StrongAI not found.")
             return

    # Run Tournament
    for opp_name, opponent in opponents_map.items():
        print(f"\nExample Match: SL Agent vs {opp_name}")
        wins = 0
        losses = 0
        draws = 0
        
        for i in range(args.games):
            render_mode = (args.games == 1)
            w, moves = play_game(agent_sl, opponent, render=render_mode) 
            if w == 1: wins += 1
            elif w == 2: losses += 1
            else: draws += 1
            print(f"Game {i+1}: {'WIN' if w==1 else 'LOSS' if w==2 else 'DRAW'} (Moves: {moves})")
            
        print(f"Results vs {opp_name}: {wins} Wins, {losses} Losses, {draws} Draws")
        if wins > losses:
            print(f"VERDICT: PASSED ({opp_name})")
        else:
            print(f"VERDICT: FAILED ({opp_name})")

if __name__ == "__main__":
    main()
