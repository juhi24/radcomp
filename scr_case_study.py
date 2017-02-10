#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vpc
from os import path

plt.ion()
plt.close('all')
np.random.seed(0)

dt0 = pd.datetime(2014, 2, 21, 12, 30)
dt1 = pd.datetime(2014, 2, 22, 15, 30)

pca, km, metadata = vpc.load_pca_kmeans('test')
fields = metadata['fields']
pn = vpc.dt2pn(dt0, dt1)
data = vpc.prepare_data(pn, fields, metadata['hmax'])
data_scaled = vpc.scale_data(data)

fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis', n_extra_ax=2)
classes = vpc.classify(data_scaled, km)

for iax in range(len(axarr)-1):
    vpc.class_colors(classes, ax=axarr[iax])

csv_path = path.join(vpc.HOME, 'results', 'pip2015', 'params.csv')
data_ground = pd.read_csv(csv_path, parse_dates=['datetime'], index_col=['winter', 'case', 'datetime'])
data_g = data_ground.loc['first','2014 Feb 21-2014 Feb 22']
#data_g.density.plot(ax=axarr[-1])
ax_rho = axarr[-2]
ax_d = axarr[-1]
ax_rho.plot(data_g['density'].index.values, data_g['density'].values, drawstyle='steps')
ax_rho.set_ylabel('Density, kg$\,$m$^{-3}$')
ax_d.plot(data_g['D_max'].index.values, data_g['D_max'].values,
          label='$D_{max}$', drawstyle='steps')
ax_d.plot(data_g['D_0_gamma'].index.values, data_g['D_0_gamma'].values,
          label='$D_0$', drawstyle='steps')
ax_d.set_ylabel('mm')
ax_d.legend()
#figs, axarrs = vpc.plot_classes(data_scaled, classes, pca.n_components)