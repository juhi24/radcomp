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

plt.ion()
plt.close('all')
np.random.seed(0)

dt0 = pd.datetime(2014, 2, 21, 19, 30)
dt1 = pd.datetime(2014, 2, 22, 15, 30)
fields = ['ZH', 'zdr', 'kdp']
hmax = 10000
n_eigens = 10
plot_components = True

pn = vpc.dt2pn(dt0, dt1)
fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis')
data = vpc.prepare_data(pn, fields, hmax)
data_scaled = vpc.scale_data(data)
pca, km = vpc.train(data_scaled, n_eigens=n_eigens)
vpc.save_pca_kmeans(pca, km, data_scaled, 'test')
if plot_components:
    learn.plot_pca_components(pca, data_scaled)