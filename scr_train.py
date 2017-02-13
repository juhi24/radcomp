#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vpc
import learn
import locale
from os import path

plt.ion()
plt.close('all')
np.random.seed(0)

plot = True

locale.setlocale(locale.LC_ALL, 'C')

case_id_fmt = lambda t: t.strftime('%b%-d').lower()

casespath = path.join('cases', 'training.csv')
dts = pd.read_csv(casespath, parse_dates=['t_start', 't_end'])
dts.index = dts['t_start'].apply(case_id_fmt)
dts.index.name = 'id'

fields = ['ZH', 'zdr', 'kdp']
hmax = 10000
n_eigens = 10
plot_components = True

pnd = {}
for row in dts.itertuples():
    pane = vpc.dt2pn(row.t_start, row.t_end)
    pnd[row.Index] = pane
    print(row.t_start)
    if plot:
        fig, axarr = vpc.plotpn(pane, fields=fields+['KDP'], cmap='viridis')
        savepath = path.join(vpc.RESULTS_DIR, 'cases', row.Index+'_gc.png')
        fig.savefig(savepath)

pn = pd.concat(pnd.values(), axis=2)

#fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis')
data = vpc.prepare_data(pn, fields, hmax)
data_scaled = vpc.scale_data(data)
pca, km = vpc.train(data_scaled, n_eigens=n_eigens)
vpc.save_pca_kmeans(pca, km, data_scaled, '2014rhi')
if plot_components:
    learn.plot_pca_components(pca, data_scaled)

window = 15
pn = pnd['mar21']

threshold = dict(zdr=4.5, kdp=0.3)
for field, data in pn.iteritems():
    if field not in threshold.keys():
        continue
    for dt, col in data.iteritems():
        winsize=1
        median_trigger = False
        threshold_trigger = False
        dat_new = col.iloc[:window].copy()
        while winsize<window:
            winsize += 1
            dat = col.iloc[:winsize].copy()
            med = dat.median()
            if med < 0.7*threshold[field]:
                break
            threshold_exceeded = dat.isnull().any() and med>threshold[field]
            median_limit_exceeded = med > 8*dat.abs().min()
            view = pn[field, :, dt].iloc[:window]
            if median_limit_exceeded:
                print(field + ', ' + str(dt) + ': median ' + str(med))
                #print(view)
                view[view>0.95*med] = np.nan
                #print(view)
                break
            if threshold_exceeded:
                print(field + ', ' + str(dt) + ': thresh')
                #print(view)
                view[view>threshold[field]] = np.nan
                #print(view)
                break
            
