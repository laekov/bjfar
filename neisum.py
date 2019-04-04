import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


n_features = 512
n_repeat = 7

p_dropout = .9


class NeiSum(nn.Module):
    def __init__(self):
        super(NeiSum, self).__init__()
        self.fc1 = nn.Linear(15, n_features)
        self.fc2 = nn.Linear(n_features * 2, n_features)
        self.fc3 = nn.Linear(n_features * 2, 1)
        self.fc4 = nn.Linear(n_features * 2, 1)
        self.bn = nn.BatchNorm1d(n_features)
        self.train_mask = None
        
    def forward(self, x_in, adjacent_matrix):
        x = torch.relu(self.fc1(x_in))
        x_init = x
        for _ in range(n_repeat):
            nei_x = torch.sparse.mm(adjacent_matrix, x)
            x = torch.cat((x, nei_x), dim=1)
            x = torch.relu(self.bn(self.fc2(x)))
        x = torch.cat((x, x_init), dim=1)
        ratio = self.fc3(x)
        delta = self.fc4(x)
        original_x = x_in[:, 0].reshape(-1, 1)
        return ratio * original_x + delta

    def loss(self, out, std, is_train=False):
        out = out.reshape_as(std)
        mask = torch.ones_like(out)
        if is_train:
            mask = F.dropout(mask, p=p_dropout)
            if self.train_mask is None:
                self.train_mask = torch.tensor([1. if i < 20000 else 0. for i in range(out.shape[0])]).cuda()
            mask *= self.train_mask
        sse = (((out - std) * mask)**2).sum()
        ss0 = (((out - out.mean()) * mask)**2).sum()
        return sse, sse / ss0
        


    def train_iter(model, optim, datas, labels, graph):
        n_iter = int(1. / (1. - p_dropout))
        losses = []
        for _ in range(n_iter):
            out = model(datas, graph)
            loss, r2 = model.loss(out, labels, is_train=True)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    def validate(model, datas, labels, graph):
        model.eval()
        out = model(datas, graph)
        loss, r2 = model.loss(out[20000:], labels[20000:])
        return loss
