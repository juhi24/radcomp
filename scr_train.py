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
        savepath = path.join(vpc.RESULTS_DIR, 'cases', row.Index+'_gcmf1.png')
        fig.savefig(savepath)

pn = pd.concat(pnd.values(), axis=2)

#fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis')
data = vpc.prepare_data(pn, fields, hmax)
data_scaled = vpc.scale_data(data)
pca, km = vpc.train(data_scaled, n_eigens=n_eigens)
vpc.save_pca_kmeans(pca, km, data_scaled, '2014rhi')
if plot_components:
    learn.plot_pca_components(pca, data_scaled)

pnt = pnd['jan31']
pnt['zdr'] = pnt['ZDR'].copy()
heigth_px=35
crop_px=20
size=(18, 3)

pn_new = pnt.copy()
ground_threshold = dict(ZDR=3.5, KDP=0.22)
keys = list(map(str.lower, ground_threshold.keys()))
pn_new = vpc.create_filtered_fields_if_missing(pn_new, keys)
for field in keys:
    view = pn_new[field].iloc[:heigth_px]
    fltrd = vpc.median_filter_df(view, param=field, fill=True,
                                 nullmask=pnt.ZH.isnull(), size=size)
    new_values = fltrd.iloc[:crop_px]
    selection = pnt[field.upper()]>ground_threshold[field.upper()]
    selection.loc[:, selection.iloc[crop_px]] = False # not clutter
    selection.loc[:, selection.iloc[0]] = True
    selection.iloc[crop_px:] = False
    df = pn_new[field].copy()
    df[selection] = new_values[selection]
    pn_new[field] = df

vpc.plotpn(pn_new, fields=['ZH', 'ZDR', 'zdr', 'KDP', 'kdp'], cmap='viridis')