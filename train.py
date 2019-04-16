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

import time

import argparse


def generate_sparse_matrix(neighbors):
    xs = []
    ys = []
    ws = []
    for i, n in enumerate(neighbors):
        idx = int(n[0])
        w = 1. / len(n[1].split(';'))
        for j in n[1].split(';'):
            if len(j) == 0:
                continue
            xs.append(idx)
            ys.append(int(float(j)))
            ws.append(w)
    return xs, ys, ws


def get_geop_cols(indices):
    global raw_data
    return np.array(geop.GeoDataFrame(raw_data[indices]))


def init_data(min_ring=0):
    global raw_data, datas, adjacent_matrix, ids, id_map

    raw_data = geop.read_file('planblocksR5_v3/planblocksR5_v3.shp')
    raw_data = raw_data[raw_data['r5id'].isin(list(range(min_ring, 6)))].reset_index()
    n = len(raw_data)

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

    edge_valid = [x in id_map and y in id_map for x, y in zip(edgex, edgey)]
    adjacent_matrix = torch.sparse.FloatTensor(
        torch.LongTensor([
                [id_map[x] for i, x in enumerate(edgex) if edge_valid[i]], 
                [id_map[y] for i, y in enumerate(edgey) if edge_valid[i]]
            ]), 
            torch.tensor(
                [w for i, w in enumerate(edgew) if edge_valid[i]],
                dtype=torch.float32),
            (n, n)).cuda()
    return raw_data


def train_model(model_name, model, optim, n):
    global datas, adjacent_matrix
    train_datas = torch.tensor(datas[:, :-2]).float().cuda()
    labels = torch.tensor(datas[:, -2]).float().cuda()
    losses = []
    min_validate = 1e10
    if load_prev_best:
        min_validate = model.validate(train_datas, labels, adjacent_matrix)
        print('Previous best loss = {}'.format(min_validate))
    no_down_cnt = 0
    for i in range(n):
        train_begin = time.time()
        loss = model.train_iter(optim, train_datas, labels, adjacent_matrix)
        losses.append(loss)
        train_end = time.time()
        if i % 10 == 0:
            loss_validate = model.validate(train_datas, labels, adjacent_matrix)
            print('Iteration {} Mean training loss = {}, Validate loss = {}, time = {} s'
                  .format(i, np.mean(losses), loss_validate, train_end - train_begin))
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
    'mlp2': MLP,
    'mlp3': MLP,
    'mlp4': MLP,
    'mlp5': MLP,
    'neisum': NeiSum,
    'neiatt': NeiAtt,
    'neiatt2': NeiAtt,
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
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('-m', '--model', default='mlp', required=True)
    parser.add_argument('-l', '--lr', type=float, default=3e-8)
    parser.add_argument('-r', '--ring', type=int, default=0)
    parser.add_argument('--load', type=int, default=1)

    args = parser.parse_args()


    lr = args.lr
    load_prev_best = args.load
    model_name = args.model
    min_ring = args.ring

    init_data(min_ring)

    print('Training mode {} with lr = {} total datas = {}'.format(model_name, lr,
                                                                  len(datas)))

    train(model_name)
