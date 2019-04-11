import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from neisum import NeiSum


n_features = 512
n_repeat = 4

p_dropout = .9


class NeiAtt(NeiSum):
    def __init__(self):
        super(NeiAtt, self).__init__()
        self.attention_fc = nn.Linear(n_features, n_features, bias=False)

    def attention(self, x, g):
        ind = g._indices()
        exp_x = x[ind[0]]
        exp_y = self.attention_fc(x)[ind[1]]
        wei = (exp_x * exp_y).sum(dim=1) * g._values()
        return torch.sparse.FloatTensor(ind, wei, g.shape)
        
    def forward(self, x_in, adjacent_matrix):
        x = torch.relu(self.fc1(x_in))
        x_init = x
        for _ in range(n_repeat):
            weights = self.attention(x, adjacent_matrix)
            nei_x = torch.sparse.mm(weights, x)

            x = torch.cat((x, nei_x), dim=1)
            x = torch.relu(self.bn(self.fc2(x)))
        x = torch.cat((x, x_init), dim=1)
        ratio = self.fc3(x)
        delta = self.fc4(x)
        original_x = x_in[:, 0].reshape(-1, 1)
        return ratio * original_x + delta
