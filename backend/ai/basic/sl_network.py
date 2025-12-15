import torch.nn as nn

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
