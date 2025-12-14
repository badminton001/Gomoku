import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.game_engine import GameEngine
from backend.algorithms.classic_ai import RandomAgent, GreedyAgent
try:
    from backend.algorithms.strong_ai import AlphaBetaAgent
except ImportError:
    AlphaBetaAgent = None

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
        for x in range(board.size):
            for y in range(board.size):
                if board.is_valid_move(x, y):
                    board.place_stone(x, y, player)
                    if board.check_five_in_a_row(player):
                        board.board[x][y] = 0
                        return x, y
                    board.board[x][y] = 0
        
        # Check for immediate loss (Opponent 4 -> 5)
        opponent = 3 - player
        for x in range(board.size):
            for y in range(board.size):
                if board.is_valid_move(x, y):
                    board.place_stone(x, y, opponent)
                    if board.check_five_in_a_row(opponent):
                        board.board[x][y] = 0
                        return x, y # Must block!
                    board.board[x][y] = 0
        
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
            return 3 - p # Opponent wins
            
    if render:
        print_board(engine.board)
        print(f"Winner: {engine.winner}")
        
    return engine.winner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", action="store_true", help="Play against the AI")
    parser.add_argument("--vs", choices=["random", "greedy", "strong"], default="strong", help="Opponent type")
    parser.add_argument("--games", type=int, default=1, help="Number of games to simulate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_sl = SLAgent("models/sl_model_v1.pth", device=device)
    
    if args.human:
        print("\n>>> HUMAN vs SL AI CHALLENGE <<<")
        print("You are Player 2 (O). AI is Player 1 (X).")
        opponent = HumanAgent()
        play_game(agent_sl, opponent, render=True)
        return

    # Select Opponent
    if args.vs == "random":
        print("Opponent: RandomAgent")
        opponent = RandomAgent()
    elif args.vs == "greedy":
        print("Opponent: GreedyAgent")
        opponent = GreedyAgent()
    elif args.vs == "strong":
        if AlphaBetaAgent:
            print("Opponent: StrongAlphaBeta (Depth 2)")
            opponent = AlphaBetaAgent(depth=2, time_limit=1.0)
        else:
            print("StrongAI not found, falling back to Greedy")
            opponent = GreedyAgent()
    
    # Run Benchmark
    print(f"\nRunning {args.games} Games: SL Agent (P1) vs {args.vs.capitalize()} (P2)")
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(args.games):
        w = play_game(agent_sl, opponent, render=False) # Only render if single game? Let's turn off for benchmark
        if w == 1: wins += 1
        elif w == 2: losses += 1
        else: draws += 1
        print(f"Game {i+1}: {'WIN' if w==1 else 'LOSS' if w==2 else 'DRAW'}")
        
    print(f"\nResults: {wins} Wins, {losses} Losses, {draws} Draws")
    if wins > losses:
        print("VERDICT: PASSED (Stronger)")
    else:
        print("VERDICT: FAILED (Weaker)")

if __name__ == "__main__":
    main()
