import numpy as np
import torch
import torch.nn as nn

def normalize(state):
    a, s, p, n, icu = state
    return np.array([
        a / 480,
        s / 480,
        p / 2.0,
        n,
        icu
    ], dtype=np.float32)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
