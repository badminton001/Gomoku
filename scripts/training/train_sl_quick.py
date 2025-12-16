import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from backend.ai.policy_network import Net

class GoMokuDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, label = self.data[idx]
        return torch.tensor(board, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

def train(dataset_path="data/anti_greedy.pkl", model_path="models/sl_policy_v1_base.pth", save_path="models/sl_policy_v1_finetuned.pth", epochs=3):
    print(f"Loading dataset from {dataset_path}...")
    dataset = GoMokuDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    
    # Load previous weights
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded base model: {model_path}")
    else:
        print("Base model not found, starting fresh.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for boards, labels in dataloader:
            boards, labels = boards.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {100*correct/total:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

if __name__ == "__main__":
    train()
