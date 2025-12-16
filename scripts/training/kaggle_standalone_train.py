import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import time
import math
import pickle
import os
from joblib import Parallel, delayed
from typing import List, Tuple, Optional, Dict

# 1. BOARD & ENGINE (Simplified)

class Board:
    def __init__(self, size: int = 15) -> None:
        self.size: int = size
        self.board: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]
        self.move_count: int = 0
        self._DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def is_empty(self, x: int, y: int) -> bool:
        return self.board[x][y] == 0
        
    def is_valid_move(self, x: int, y: int) -> bool:
        return self.is_inside(x, y) and self.is_empty(x, y)

    def place_stone(self, x: int, y: int, player: int) -> bool:
        if not self.is_valid_move(x, y): return False
        self.board[x][y] = player
        self.move_count += 1
        return True

    def check_five_in_a_row(self, player: int) -> bool:
        n = self.size
        for x in range(n):
            for y in range(n):
                if self.board[x][y] != player: continue
                # Optimized check
                board_val = self.board
                for dx, dy in self._DIRS:
                    c = 1
                    cx, cy = x + dx, y + dy
                    while 0 <= cx < n and 0 <= cy < n and board_val[cx][cy] == player:
                        c += 1; cx += dx; cy += dy
                    if c >= 5: return True
        return False

    def get_game_result(self):
        if self.check_five_in_a_row(1): return 1
        if self.check_five_in_a_row(2): return 2
        if self.move_count >= self.size * self.size: return 3
        return 0

class GameEngine:
    def __init__(self):
        self.board = Board()
        self.current_player = 1
        self.winner = 0
        self.game_over = False

    def make_move(self, x, y):
        if self.game_over: return False
        if self.board.place_stone(x, y, self.current_player):
            res = self.board.get_game_result()
            if res != 0:
                self.winner = res
                self.game_over = True
            else:
                self.current_player = 3 - self.current_player
            return True
        return False

def get_neighbor_moves(board, distance=2):
    moves = set()
    size = board.size
    b = board.board
    for x in range(size):
        for y in range(size):
            if b[x][y] != 0:
                for dx in range(-distance, distance + 1):
                    for dy in range(-distance, distance + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size and b[nx][ny] == 0:
                            moves.add((nx, ny))
    return list(moves)

# 2. AI AGENTS

SCORE_FIVE = 10000000
SCORE_LIVE_4 = 1000000
SCORE_DEAD_4 = 100000
SCORE_LIVE_3 = 100000
SCORE_DEAD_3 = 1000

class AlphaBetaAgent:
    def __init__(self, depth: int = 2, time_limit: float = 1.0):
        self.depth = depth
        self.time_limit = time_limit

    def get_move(self, board: Board, player: int) -> Tuple[int, int]:
        candidates = self.get_sorted_moves(board, player)
        if not candidates: return (7, 7)
        if len(candidates) == 1: return candidates[0]
        
        best_val = -math.inf
        best_move = candidates[0]
        alpha = -math.inf
        beta = math.inf
        
        for mx, my in candidates:
            board.place_stone(mx, my, player)
            val = -self.negamax(board, self.depth - 1, -beta, -alpha, 3 - player)
            board.board[mx][my] = 0
            board.move_count -= 1
            
            if val > best_val:
                best_val = val
                best_move = (mx, my)
            alpha = max(alpha, best_val)
            if alpha >= beta: break # Root pruning
        return best_move

    def negamax(self, board, depth, alpha, beta, color) -> float:
        res = board.get_game_result()
        if res != 0:
            if res == color: return SCORE_FIVE
            if res == (3-color): return -SCORE_FIVE
            return 0
        if depth == 0:
            return self.evaluate_shape(board, color)
            
        candidates = self.get_sorted_moves(board, color)
        if not candidates: return 0
        
        value = -math.inf
        for mx, my in candidates:
            # Removed time constraint for Kaggle batch processing
            board.place_stone(mx, my, color)
            val = -self.negamax(board, depth - 1, -beta, -alpha, 3 - color)
            board.board[mx][my] = 0
            board.move_count -= 1
            value = max(value, val)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value

    def get_sorted_moves(self, board, player):
        moves = get_neighbor_moves(board, 2)
        if not moves: return [(7,7)]
        scored = []
        opp = 3 - player
        for mx, my in moves:
            # Heuristic embedded
            board.board[mx][my] = player
            atk = self._quick_score(board, mx, my, player)
            board.board[mx][my] = 0
            
            board.board[mx][my] = opp
            dfs = self._quick_score(board, mx, my, opp)
            board.board[mx][my] = 0
            
            score = 0
            if dfs >= SCORE_FIVE: score = SCORE_FIVE * 2
            elif atk >= SCORE_FIVE: score = SCORE_FIVE
            elif dfs >= SCORE_LIVE_4: score = SCORE_LIVE_4 * 2
            elif atk >= SCORE_LIVE_4: score = SCORE_LIVE_4
            elif dfs >= SCORE_DEAD_4: score = SCORE_DEAD_4 * 2 # Block 4
            elif atk >= SCORE_DEAD_4: score = SCORE_DEAD_4
            else: score = atk + dfs
            scored.append((score, (mx, my)))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:15]] # Prune to top 15 for speed

    def _quick_score(self, board, x, y, player):
        score = 0
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        size = board.size
        b = board.board
        for dx, dy in dirs:
            c = 1
            nx, ny = x+dx, y+dy
            while 0<=nx<size and 0<=ny<size and b[nx][ny] == player: c+=1; nx+=dx; ny+=dy
            nx, ny = x-dx, y-dy
            while 0<=nx<size and 0<=ny<size and b[nx][ny] == player: c+=1; nx-=dx; ny-=dy
            if c >= 5: score += SCORE_FIVE
            elif c == 4: score += SCORE_LIVE_4
            elif c == 3: score += SCORE_LIVE_3
            elif c == 2: score += 100
        return score

    def evaluate_shape(self, board, player):
        score = 0
        opp = 3 - player
        for r in range(15): score += self.eval_line(board.board[r], player, opp)
        for c in range(15): score += self.eval_line([board.board[r][c] for r in range(15)], player, opp)
        return score

    def eval_line(self, line, p, o):
        s = "".join(str(x) for x in line)
        score = 0
        ps, os_str = str(p), str(o)
        if ps*5 in s: score += SCORE_FIVE
        if "0"+ps*4+"0" in s: score += SCORE_LIVE_4
        if ps+"0"+ps*3 in s or ps*3+"0"+ps in s: score += SCORE_DEAD_4
        if "0"+ps*3+"0" in s: score += SCORE_LIVE_3
        
        opp_score = 0
        if os_str*5 in s: opp_score += SCORE_FIVE
        if "0"+os_str*4+"0" in s: opp_score += SCORE_LIVE_4
        if os_str+"0"+os_str*3 in s: opp_score += SCORE_DEAD_4
        
        return score - opp_score * 1.5

class GreedyAgent:
    def get_move(self, board, player):
        moves = get_neighbor_moves(board, 2)
        if not moves: return (7,7)
        return random.choice(moves)

class RandomAgent:
    def get_move(self, board, player):
        moves = get_neighbor_moves(board, 2)
        if not moves: return (7,7)
        return random.choice(moves)

# 3. PARALLEL DATA GENERATION

def play_one_game(teacher_depth, opponent_type):
    # worker function for Parallel
    teacher = AlphaBetaAgent(depth=teacher_depth)
    
    if opponent_type == "Random":
        opponent = RandomAgent()
    elif opponent_type == "Greedy":
        opponent = GreedyAgent()
    else:
        opponent = AlphaBetaAgent(depth=teacher_depth) # Strong
        
    engine = GameEngine()
    engine.make_move(7, 7) # Open
    
    # 50% chance to random 2nd move
    if random.random() < 0.5:
        try:
             moves = get_neighbor_moves(engine.board)
             rx, ry = random.choice(moves)
             engine.make_move(rx, ry) 
        except: pass
        
    game_history = []
    # Randomize who is teacher (P1 or P2)
    teacher_p = 1 if random.random() < 0.5 else 2
    
    while not engine.game_over and engine.board.move_count < 225:
        p = engine.current_player
        
        # Snapshot input
        mnode = np.zeros((15,15), dtype=np.float32)
        for r in range(15):
            for c in range(15):
                v = engine.board.board[r][c]
                if v == p: mnode[r][c] = 1
                elif v != 0: mnode[r][c] = -1
                
        if p == teacher_p:
            mx, my = teacher.get_move(engine.board, p)
            game_history.append((p, mnode, (mx, my)))
        else:
            mx, my = opponent.get_move(engine.board, p)
            
        if not engine.make_move(mx, my): break
        
    samples = []
    if engine.winner == teacher_p:
        for pl, inp, (mx, my) in game_history:
             if pl == teacher_p:
                 label = mx * 15 + my
                 samples.append((inp, label))
    return samples

def generate_parallel(total_games=1000):
    print(f"Generating Parallel Dataset ({total_games} games)...")
    start = time.time()
    
    n_jobs = 4 # Kaggle has 4 cores usually
    
    tasks = []
    # Phase 1: Random
    tasks.extend([("Random") for _ in range(200)])
    # Phase 2: Greedy
    tasks.extend([("Greedy") for _ in range(300)])
    # Phase 3: Strong (Reduced count but Parallel helps)
    tasks.extend([("Strong") for _ in range(500)])
    
    # Execute
    print(f"Starting {len(tasks)} games on {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(delayed(play_one_game)(2, t) for t in tasks)
    
    dataset = []
    for res in results:
        dataset.extend(res)
        
    print(f"Generation Complete. {len(dataset)} samples. Time: {time.time()-start:.1f}s")
    
    # Save dataset for download
    with open("gomoku_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("Saved 'gomoku_dataset.pkl'. You can download this file now.")
    
    return dataset

# 4. TRAINING

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

class AugmentedGomokuData(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data) * 8 
    def __getitem__(self, idx):
        real_idx = idx // 8
        sym_idx = idx % 8
        b, l = self.data[real_idx]
        lx, ly = divmod(l, 15)
        tb = b.copy()
        tx, ty = lx, ly
        
        if sym_idx >= 4:
            tb = np.fliplr(tb)
            ty = 14 - ty
            sym_idx -= 4
        for _ in range(sym_idx):
            tb = np.rot90(tb)
            tx, ty = 14-ty, tx
        
        return torch.tensor(tb.copy()).unsqueeze(0), torch.tensor(tx*15+ty, dtype=torch.long)

def train(dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(AugmentedGomokuData(dataset), batch_size=256, shuffle=True, num_workers=2)
    
    model.train()
    for ep in range(30):
        loss_avg = 0
        correct = 0; total = 0
        st = time.time()
        for b, l in loader:
            b, l = b.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(b)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            _, p = torch.max(out, 1)
            correct += (p==l).sum().item()
            total += l.size(0)
        
        avg = loss_avg/len(loader)
        acc = 100*correct/total
        print(f"Epoch {ep+1} | Loss: {avg:.4f} | Acc: {acc:.2f}% | {time.time()-st:.1f}s")
        scheduler.step(avg)
        
    torch.save(model.state_dict(), "sl_model_kaggle.pth")
    print("Saved 'sl_model_kaggle.pth'.")

if __name__ == "__main__":
    data = generate_parallel(total_games=1000)
    train(data)
