#!/usr/bin/env python
# coding: utf-8
import geopandas as geop
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from mlp import MLP
from neisum import NeiSum
from neiatt import NeiAtt

raw_data = geop.read_file('./PlanBlocksR5_v2neighbors.shp')

lr = 1e-6
load_prev_best = False # True


def generate_sparse_matrix(neighbors):
    id_map = dict()
    xs = []
    ys = []
    ws = []
    for i, n in enumerate(neighbors):
        idx = int(n[0])
        id_map[idx] = i
        w = 1. / len(n[1].split(';'))
        for j in n[1].split(';'):
            if len(j) == 0:
                continue
            xs.append(idx)
            ys.append(int(float(j)))
            ws.append(w)
    return xs, ys, ws



def get_geop_cols(indices):
    return np.array(geop.GeoDataFrame(raw_data[indices]))


def init_data():
    global datas, adjacent_matrix, ids, id_map

    neighbors = np.array(geop.GeoDataFrame(raw_data[['Block_ID', 'neighbors']]))
    ids = get_geop_cols('Block_ID')
    fars = get_geop_cols(['Far_2004', 'Far_2017'])
    distances = get_geop_cols(['d_tam', 'd_metro', 'd_cbd', 'd_zgc'])
    ranges = get_geop_cols(['in_Eco', 'in_Ind', 'in_His'])
    popu = get_geop_cols(['Density00', 'Density10', 'Density22'])
    usage = get_geop_cols(['Perc_Res', 'Perc_Job'])
    misc = get_geop_cols(['Den_road', 'Area_m2'])
    d_metro = np.exp(- distances[:, 1]).reshape(-1, 1)
    d_other = np.exp(- distances[:, [0, 2, 3]] * .05)
    d_dis = np.concatenate((d_metro, d_other), axis=1)
    d_popu = np.log(popu + 1.)
    d_popu /= d_popu.max()
    d_road = np.log(misc[:, 0] + 1.)
    d_road /= d_road.max()
    d_area = np.log(misc[:, 1] + 1.)
    d_area /= d_area.max()
    datas = np.concatenate((fars[:, 0].reshape(-1, 1),
                            d_dis,
                            ranges,
                            d_popu,
                            usage,
                            d_road.reshape(-1, 1),
                            d_area.reshape(-1, 1),
                            fars[:, 1].reshape(-1, 1),
                            ids.reshape(-1, 1)), axis=1)
    np.random.seed(1234324)
    np.random.shuffle(datas)

    edgex, edgey, edgew = generate_sparse_matrix(neighbors)

    id_map = dict()
    for i, blk in enumerate(datas[:, -1]):
        id_map[int(blk)] = i

    n = datas.shape[0]
    adjacent_matrix = torch.sparse.FloatTensor(
        torch.LongTensor([[id_map[x] for x in edgex], [id_map[y] for y in edgey]]),
        torch.tensor(np.array(edgew), dtype=torch.float32), (n, n)).cuda()


def train_model(model_name, model, optim, n):
    global datas, adjacent_matrix
    train_datas = torch.tensor(datas[:, :-2]).float().cuda()
    labels = torch.tensor(datas[:, -2]).float().cuda()
    losses = []
    min_validate = 1e10
    no_down_cnt = 0
    for i in range(n):
        loss = model.train_iter(optim, train_datas, labels, adjacent_matrix)
        losses.append(loss)
        if i % 10 == 0:
            loss_validate = model.validate(train_datas, labels, adjacent_matrix)
            print('Iteration {} Mean training loss = {}, Validate loss = {}'
                  .format(i, np.mean(losses), loss_validate))
            if loss_validate < min_validate:
                torch.save(model.state_dict(), 'best_{}.pt'.format(model_name))
                min_validate = loss_validate
                no_down_cnt = 0
            else:
                no_down_cnt += 1
            losses = []
    model.load_state_dict(torch.load('best_{}.pt'.format(model_name)))
    loss_validate = model.validate(train_datas, labels, adjacent_matrix)
    print('Best validation loss = {}'.format(loss_validate))


models = {
    'mlp': MLP,
    'neisum': NeiSum,
    'neiatt': NeiAtt,
}


def train(model_name='mlp'):
    model = models[model_name]()
    if load_prev_best:
        model.load_state_dict(torch.load('best_{}.pt'.format(model_name)))
    model.cuda()
    optim1 = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model_name, model, optim1, 200000)


def predict(model_name='mlp'):
    global datas, adjacent_matrix, ids, id_map
    model = models[model_name]()
    model.load_state_dict(torch.load('best_{}.pt'.format(model_name)))
    model.cuda().eval()
    prediction_tensor = model(torch.tensor(datas[:, :-2]).float().cuda(), adjacent_matrix)
    predictions = prediction_tensor.cpu().detach().numpy().reshape(-1)
    preds = [predictions[id_map[int(li)]] for li in ids]
    labels = [datas[id_map[int(li)], -2] for li in ids]
    return preds, labels


if __name__ == '__main__':
    init_data()
    train('neiatt')
