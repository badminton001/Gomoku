import torch
import torch.nn as nn
import numpy as np
import sys
import os

# --- DEFINE NETWORK (Must match training) ---
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

def debug_state():
    print(">>> LOADING MODEL DIAGNOSTIC <<<")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = Net().to(device)
        model.load_state_dict(torch.load("models/sl_model_v1.pth", map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # Reconstruct the board where AI lost
    # Based on user logs: O (P2) has vertical line at Col 6.
    # Let's verify a state where O has 3 stones (Live 3) and it's X's turn to block.
    # O: (3,6), (4,6), (5,6)
    # X: (2,5), (2,6), (3,4) (Guessing from board)
    
    board = np.zeros((15, 15), dtype=int)
    
    # Setup stones (1=AI/Black, 2=Opponent/White)
    # O stones
    board[3][6] = 2
    board[4][6] = 2
    board[5][6] = 2
    
    # X stones (Simulated)
    board[2][5] = 1
    board[2][6] = 1 # Blocked top
    board[3][4] = 1
    
    print("\n--- RECONSTRUCTED BOARD (X to move) ---")
    # Expected: X MUST block at (6,6) to stop the 3->4 advancement
    
    # Visualize
    print("   0 1 2 3 4 5 6 7 8 ...")
    for r in range(10):
        row = f"{r}  "
        for c in range(10):
            if board[r][c] == 1: row += "X "
            elif board[r][c] == 2: row += "O "
            else: row += ". "
        print(row)
        
    # Prepare Input
    inp = np.zeros((15, 15), dtype=np.float32)
    player = 1 # AI to move
    for r in range(15):
        for c in range(15):
            v = board[r][c]
            if v == player: inp[r][c] = 1.0
            elif v != 0: inp[r][c] = -1.0
            
    t_inp = torch.tensor(inp).unsqueeze(0).unsqueeze(0).to(device)
    
    print("\n>>> QUERIES NEURAL NETWORK <<<")
    with torch.no_grad():
        logits = model(t_inp)
        probs = torch.softmax(logits, dim=1)
        # Get Top 10
        sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]
        
    print("Top 10 Recommend Moves:")
    for i in range(10):
        idx = sorted_indices[i].item()
        p = probs[0][idx].item()
        x, y = divmod(idx, 15)
        print(f"Rank {i+1}: ({x}, {y}) - Conf: {p:.4f}")
        
    # Check if (6,6) is in top candidates
    target = (6,6)
    found = False
    for i in range(10):
        idx = sorted_indices[i].item()
        x, y = divmod(idx, 15)
        if (x, y) == target:
            print(f"\nSUCCESS: Critical Block {target} found at Rank {i+1}")
            found = True
            break
            
    if not found:
        print(f"\nFAILURE: Critical Block {target} NOT in Top 10.")
        print("CONCLUSION: The model is ignoring the threat. Likely untrained or bad data.")

if __name__ == "__main__":
    debug_state()
