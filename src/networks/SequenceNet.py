import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_net import BaseNet

class SequenceNet(BaseNet):
    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.lstm = nn.LSTM(input_size=384, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # take the output from the last time step
        x = self.fc1(x)
        return x


class SequenceAutoencoder(BaseNet):
    def __init__(self):
        super().__init__()

        self.rep_dim = 128

        # Encoder (must match the Deep SVDD network above)
        self.lstm = nn.LSTM(input_size=384, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.fc2 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(128, 384, bias=False)

    def forward(self, x, debug=False):
        if debug: print(f"Input shape: {x.shape}")
        x, _ = self.lstm(x)
        if debug: print(f"After lstm: {x.shape}")
        x = x[:, -1, :]  # take the output from the last time step
        if debug: print(f"After selecting last time step: {x.shape}")
        x = self.bn1d(self.fc1(x))
        if debug: print(f"After fc1 and bn1d: {x.shape}")
        x = F.leaky_relu(x)
        x = self.bn1d2(self.fc2(x))
        if debug: print(f"After fc2 and bn1d2: {x.shape}")
        x = self.fc3(x)
        if debug: print(f"After fc3: {x.shape}")
        return x
