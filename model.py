import torch, torch.nn as nn
from typing import OrderedDict

class IDS_NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden=128, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden//2, hidden//4), nn.BatchNorm1d(hidden//4), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden//4, 2)
        )
    def forward(self, x): return self.net(x)
    def get_weights(self): return self.state_dict()
    def set_weights(self, sd): self.load_state_dict(sd)
