import geopandas as geop

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import seaborn as sns

from scipy import stats
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


from train import init_data, train, predict


class Visualize(object):
    def __init__(self, model_name, minring=0, raw_data=None):
        levels = [0., .4, .6, 1., 2.4, 4., 6.]
        raw_data = init_data(minring, data=raw_data)

        years = []
        pred, labels = predict(model_name, years)
        predictions = np.array(pred)
        if len(years) > 0:
            y = dict({})
            for i in range(len(years)):
                pi = np.digitize(years[i], levels)
                y['d_year_{}'.format(i)] = pi
            raw_data = raw_data.join(geop.GeoDataFrame(y))
        self.preds = predictions
        labels = np.array(labels)
        self.label = labels

        self.r2 = stats.linregress(predictions, labels)[2]** 2
        
        self.error = labels - predictions
        self.d_label = np.digitize(labels, levels)
        self.d_preds = np.digitize(predictions, levels)
        
        prediction_df = geop.GeoDataFrame({
            'error': self.error,
            'lvlerror': (self.d_label - self.d_preds),
            'predictions': self.preds,
            'lvl_preds': self.d_preds,
            'lvl_truth': self.d_label,
        })

        self.df = raw_data.join(prediction_df)
    
    def plot_map(self, col):
        ax = self.df.plot(column=col, vmin=0, vmax=5 if col[0:3] != 'lvl' else 7,
                          figsize=(16, 9), legend=True, cmap='BuPu')
        ax.set_title(col)

    def plot_whole_lvl_error(self):
        # Plot error level graph
        ax = self.df.plot(column='lvlerror', vmin=-6, vmax=6, figsize=(16, 9), legend=True, cmap='coolwarm')
        ax.set_title('Level difference 2017')
    
    def plot_district(self, coors, colname='error', title='Detailed level difference 2017', vmin=-6, vmax=6, cmap='coolwarm'):
        if coors is not None:
            xs, ys = coors
            # Paint subplot in wangjing
            corners = [(xs[0], ys[0]), (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[0])]
            wangjing = geop.GeoSeries([Polygon(corners)])
            wangjing = geop.GeoDataFrame({'geometry': wangjing})

            res = geop.overlay(self.df, wangjing, how='intersection')
        else:
            res = self.df

        ax = res.plot(column=colname, vmin=vmin, vmax=vmax, figsize=(16, 9), legend=True, cmap=cmap)
        ax.set_title(title)

    def statistics(self):
        print('R^2 = {}'.format(self.r2))
        print('Level accuracy = {}'.format(np.mean(self.d_preds == self.d_label)))
        print('Mean error = {}'.format(np.mean(np.abs(self.error))))
        print('Mean level difference = {}'.format(np.abs(np.mean((self.d_label == self.d_preds)))))

    def tam(self):
        dn = np.array(self.df[['d_tam', 'error']])
        d_tam, error = dn[:, 0], dn[:, 1]
        d_tam = np.digitize(d_tam, np.arange(0, 10, 1.))
        xls = np.arange(0, 11, 1.)
        yls = [np.mean([error[i] for i in range(len(d_tam)) if d_tam[i] == lvl]) for lvl in xls]
        return plt.plot(xls, yls)

    def regression(self):
        plt.scatter(predictions, labels)
        
    def error_plot(self):
        d_error = np.array(self.df['lvlerror']).reshape(-1)
        sns.distplot(d_error, kde=False)
        


wangjing = ((116.43, 116.50), (39.97, 40.02))
w_zgc = ((116.29, 116.315), (39.97, 39.99))
        
