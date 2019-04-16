import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


n_features = 256
n_repeat = 2


batch_size = 64


def get_divide_idx(datas):
    n = len(datas)
    p = (n - 1000) // batch_size * batch_size
    return p


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(15, n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.bn = nn.BatchNorm1d(n_features)
        self.fc3 = nn.Linear(n_features, 1)
        
    def forward(self, x_in, adjacent_matrix=None):
        x = torch.relu(self.fc1(x_in))
        for _ in range(n_repeat):
            x = torch.relu(self.bn(self.fc2(x)))
        x = self.fc3(x)
        x += x_in[:, 0].reshape(-1, 1)
        return x

    def loss(self, out, std):
        out = out.reshape(-1)
        sse = ((out - std)**2).sum()
        # ss0 = ((out - out.mean())**2).sum()
        return sse

    def train_iter(model, optim, datas, labels, graph):
        di = get_divide_idx(datas)
        model.train()
        train_data = datas[:di]
        # np.random.shuffle(train_data)
        losses = []
        for i in range(0, di, batch_size):
            data = train_data[i:i+batch_size]
            label = labels[i:i+batch_size]
            out = model(data, graph)
            loss = model.loss(out, label)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    def validate(model, datas, labels, graph):
        di = get_divide_idx(datas)
        valid_data, valid_label = datas[di:], labels[di:]
        model.eval()
        out = model(valid_data, graph)
        loss = model.loss(out, valid_label)
        return loss

