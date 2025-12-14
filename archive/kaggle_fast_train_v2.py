
import os
import sys
import random
import time
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional

# ==========================================
# 1. BOARD & ENGINE
# ==========================================
class Board:
    def __init__(self, size: int = 15) -> None:
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
        self.move_count = 0

    def is_valid_move(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0

    def place_stone(self, x: int, y: int, player: int) -> bool:
        if not self.is_valid_move(x, y): return False
        self.board[x][y] = player
        self.move_count += 1
        return True

    def check_five_in_a_row(self, player: int) -> bool:
        n = self.size
        # Horizontal
        for x in range(n - 4):
            for y in range(n):
                if all(self.board[x+i][y] == player for i in range(5)): return True
        # Vertical
        for x in range(n):
            for y in range(n - 4):
                if all(self.board[x][y+i] == player for i in range(5)): return True
        # Diagonals
        for x in range(n - 4):
            for y in range(n - 4):
                if all(self.board[x+i][y+i] == player for i in range(5)): return True
                if all(self.board[x+i][y+4-i] == player for i in range(5)): return True
        return False

    def is_full(self) -> bool:
        return self.move_count >= self.size * self.size

    def get_game_result(self) -> int:
        if self.check_five_in_a_row(1): return 1
        if self.check_five_in_a_row(2): return 2
        if self.is_full(): return 3
        return 0

class GameEngine:
    def __init__(self, size: int = 15):
        self.board = Board(size=size)
        self.current_player = 1
        self.game_over = False
        self.winner = 0

    def make_move(self, x: int, y: int) -> bool:
        if self.game_over: return False
        if not self.board.place_stone(x, y, self.current_player): return False
        res = self.board.get_game_result()
        if res != 0:
            self.game_over = True
            self.winner = res
        else:
            self.current_player = 3 - self.current_player
        return True

# ==========================================
# 2. STRONG TEACHER AI (AlphaBeta)
# ==========================================
SCORE_FIVE = 10000000
SCORE_LIVE_4 = 1000000
SCORE_DEAD_4 = 100000
SCORE_LIVE_3 = 100000
SCORE_DEAD_3 = 1000
SCORE_LIVE_2 = 100
SCORE_DEAD_2 = 10

class AlphaBetaAgent:
    def __init__(self, depth: int = 2, time_limit: float = 1.0):
        self.depth = depth
        self.time_limit = time_limit
        self.start_time = 0
        self.nodes_explored = 0

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        self.start_time = time.time()
        self.nodes_explored = 0
        if board.move_count == 0: return (board.size // 2, board.size // 2)
        try:
            val, move = self.alpha_beta_search(board, player, depth=self.depth, alpha=-math.inf, beta=math.inf)
            return move
        except TimeoutError:
            return (board.size // 2, board.size // 2)

    def alpha_beta_search(self, board, player, depth, alpha, beta):
        candidates = self.get_sorted_moves(board, player)
        if not candidates: return 0, (-1, -1)
        best_val = -math.inf
        best_move = candidates[0]
        
        for mx, my in candidates:
            board.place_stone(mx, my, player)
            if board.check_five_in_a_row(player):
                board.board[mx][my] = 0
                return SCORE_FIVE, (mx, my)
            
            val = -self.negamax(board, depth - 1, -beta, -alpha, 3 - player)
            board.board[mx][my] = 0
            
            if val > best_val:
                best_val = val
                best_move = (mx, my)
            alpha = max(alpha, best_val)
            if alpha >= beta: break
        return best_val, best_move

    def negamax(self, board, depth, alpha, beta, color) -> float:
        if (self.nodes_explored & 1023) == 0:
             if time.time() - self.start_time > self.time_limit: raise TimeoutError()
        self.nodes_explored += 1
        
        res = board.get_game_result()
        if res != 0:
            if res == color: return SCORE_FIVE * (1 + depth)
            if res == (3-color): return -SCORE_FIVE * (1 + depth)
            return 0
            
        if depth == 0: return self.evaluate_shape(board, color)
        
        candidates = self.get_sorted_moves(board, color)
        if not candidates: return 0
        
        value = -math.inf
        for mx, my in candidates:
            board.place_stone(mx, my, color)
            val = -self.negamax(board, depth - 1, -beta, -alpha, 3 - color)
            board.board[mx][my] = 0
            value = max(value, val)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value

    def get_sorted_moves(self, board, player):
        moves = set()
        for x in range(board.size):
             for y in range(board.size):
                  if board.board[x][y] != 0:
                       for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                 if board.is_valid_move(x+dx, y+dy): moves.add((x+dx, y+dy))
        c = list(moves)
        if not c: return [(board.size//2, board.size//2)]
        return c[:20]

    def evaluate_shape(self, board, player):
        score = 0
        opp = 3 - player
        # Full scan
        for x in range(board.size): score += self.evaluate_line(board.board[x], player, opp)
        for y in range(board.size): score += self.evaluate_line([board.board[x][y] for x in range(board.size)], player, opp)
        for k in range(-(board.size-1), board.size):
            line = []
            for x in range(board.size):
                y = x - k
                if 0 <= y < board.size: line.append(board.board[x][y])
            if len(line)>=5: score += self.evaluate_line(line, player, opp)
        for k in range(2*board.size-1):
            line = []
            for x in range(board.size):
                y = k - x
                if 0 <= y < board.size: line.append(board.board[x][y])
            if len(line)>=5: score += self.evaluate_line(line, player, opp)
        return score

    def evaluate_line(self, line, player, opponent):
        s = "".join(str(x) for x in line)
        p, o = str(player), str(opponent)
        score = 0
        score += s.count(p*5)*SCORE_FIVE
        score += s.count("0"+p*4+"0")*SCORE_LIVE_4
        score += s.count("0"+p*4+o)*SCORE_DEAD_4
        score += s.count(o+p*4+"0")*SCORE_DEAD_4
        score += s.count("0"+p*3+"0")*SCORE_LIVE_3
        score -= s.count(o*5)*SCORE_FIVE*1.2
        score -= s.count("0"+o*4+"0")*SCORE_LIVE_4*1.2
        score -= s.count("0"+o*3+"0")*SCORE_LIVE_3*1.5
        return score

# ==========================================
# 3. DATA GENERATION & TRAINING
# ==========================================
class GDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        b, m = self.data[idx]
        return torch.tensor(b).unsqueeze(0), torch.tensor(m, dtype=torch.long)

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

def main():
    print(">>> Starting Kaggle Fast Train V2 (Monolithic Script)")
    
    # 1. Generate Data
    NUM_GAMES = 2000
    dataset = []
    teacher = AlphaBetaAgent(depth=2, time_limit=1.5) # Balanced teacher
    
    print(f"Generating {NUM_GAMES} games...")
    start_gen = time.time()
    for i in range(NUM_GAMES):
        engine = GameEngine()
        engine.make_move(7, 7)
        # Random 2nd move
        rx, ry = random.randint(6,8), random.randint(6,8)
        if engine.board.is_valid_move(rx, ry): engine.make_move(rx, ry)
        
        while not engine.game_over and engine.board.move_count < 150:
            p = engine.current_player
            inp = np.zeros((15, 15), dtype=np.float32)
            for r in range(15):
                for c in range(15):
                    v = engine.board.board[r][c]
                    if v == p: inp[r][c] = 1.0
                    elif v != 0: inp[r][c] = -1.0
            
            move = teacher.get_move(engine.board, p)
            if move == (-1, -1): break
            dataset.append((inp, move[0]*15 + move[1]))
            engine.make_move(move[0], move[1])
        
        if i % 50 == 0: print(f"Game {i} done. Samples: {len(dataset)}")
    
    print(f"Generation took {time.time()-start_gen:.1f}s. Total Samples: {len(dataset)}")
    
    # 2. Train
    print("Training Student Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    train_loader = DataLoader(GDataset(dataset), batch_size=256, shuffle=True)
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in range(50): # 50 Epochs for solid convergence
        total_loss = 0
        model.train()
        for b, m in train_loader:
            opt.zero_grad()
            out = model(b.to(device))
            loss = loss_fn(out, m.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}: Loss {total_loss/len(train_loader):.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sl_model_v1.pth")
    print("Saved models/sl_model_v1.pth")

if __name__ == "__main__":
    main()
