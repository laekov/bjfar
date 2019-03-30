import torch
import torch.nn as nn
import torch.nn.functional as F


n_features = 256
n_repeat = 3


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(15, n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.bn = nn.BatchNorm1d(n_features)
        self.fc3 = nn.Linear(n_features, 1)
        
    def forward(self, x_in):
        x = torch.relu(self.fc1(x_in))
        for _ in range(n_repeat):
            x = torch.relu(self.bn(self.fc2(x)))
        x = self.fc3(x)
        x += x_in[:, 0].reshape(-1, 1)
        return x
