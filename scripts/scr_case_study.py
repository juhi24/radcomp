#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radcomp import vpc
from os import path

plt.ion()
plt.close('all')
np.random.seed(0)

dt0 = pd.datetime(2014, 2, 21, 12, 30)
dt1 = pd.datetime(2014, 2, 22, 15, 30)

n_comp = 20

def plot_data(data, ax, **kws):
    return ax.plot(data.index, data.values, drawstyle='steps', **kws)

def fltr_long_period(data_in, minutes=30):
    data = data_in.copy()
    selection = data.tdelta>pd.tseries.offsets.Minute(minutes).delta
    data.loc[selection,['density','D_max','D_0_gamma', 'N_w']] = np.nan
    return data

pca, km, metadata = vpc.load_pca_kmeans('2014rhi_{n}comp'.format(n=n_comp))
fields = metadata['fields']
pn = vpc.dt2pn(dt0, dt1)
data = vpc.prepare_data(pn, fields, metadata['hmax'])
data_scaled = vpc.scale_data(data)

fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis', n_extra_ax=3)
classes = vpc.classify(data_scaled, km)

for iax in range(len(axarr)-1):
    vpc.class_colors(classes, ax=axarr[iax])

csv_path = path.join(vpc.HOME, 'results', 'pip2015', 'params.csv')
data_ground = pd.read_csv(csv_path, parse_dates=['datetime'], index_col=['winter', 'case', 'datetime'])
data_g = data_ground.loc['first','2014 Feb 21-2014 Feb 22']
#data_g = fltr_long_period(data_g) # TODO
#data_g.density.plot(ax=axarr[-1])
ax_r = axarr[-3]
ax_rho = axarr[-2]
ax_d = axarr[-1]
plot_data(data_g['intensity'], ax_r)
ax_r.set_ylabel('LWE, mm$\,$h$^{-1}$')
plot_data(data_g['density'], ax_rho)
ax_rho.set_ylabel('Density, kg$\,$m$^{-3}$')
plot_data(data_g['D_max'], ax_d, label='$D_{max}$')
plot_data(data_g['D_0_gamma'], ax_d, label='$D_0$')
ax_d.set_ylabel('mm')
ax_d.legend()
#figs, axarrs = vpc.plot_classes(data_scaled, classes, pca.n_components)