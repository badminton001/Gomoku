import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------------------------------------------------------
# Neural Network Architecture
# -------------------------------------------------------------------------
class GomokuNet(nn.Module):
    def __init__(self, board_size=15):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # Input: 1 channel (board state: 1=me, -1=opp, 0=empty)
        # Or 2 channels? Let's stick to 1 float channel for simplicity like AlphaGo Zero used (mostly)
        # Better: 3 channels (My stones, Opponent stones, Color to play)
        # But our generation script saved a single 15x15 matrix with 1/-1.
        # We will usage that.
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # Reduce to 2 filters
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
    
    def forward(self, x):
        # x: (Batch, 15, 15) -> (Batch, 1, 15, 15)
        x = x.unsqueeze(1) 
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = self.relu(self.policy_bn(self.policy_conv(x)))
        x = x.view(x.size(0), -1)
        x = self.policy_fc(x)
        return x # Logits

# -------------------------------------------------------------------------
# Dataset Loader
# -------------------------------------------------------------------------
class GomokuDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f) # List of (board, move)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, move = self.data[idx]
        # Data augmentation could go here (rotate/flip)
        # For "Fast", we skip augmentation for now
        return torch.tensor(board, dtype=torch.float32), torch.tensor(move, dtype=torch.long)

# -------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------
def train_sl(data_path="data/sl_dataset_v1.pkl", save_path="models/sl_gomoku.pth", epochs=20):
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Run generate_dataset.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    
    dataset = GomokuDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = GomokuNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for boards, moves in dataloader:
            boards, moves = boards.to(device), moves.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += moves.size(0)
            correct += (predicted == moves).sum().item()
            
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")
        
    # Save
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_sl(epochs=30) # 30 epochs is better for 300 games
