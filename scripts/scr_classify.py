#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radcomp import vpc

plt.ion()
plt.close('all')
np.random.seed(0)

dt0 = pd.datetime(2014, 2, 21, 12, 30)
dt1 = pd.datetime(2014, 2, 22, 15, 30)


def plot_case(dt0, dt1, n_comp=20):
    pca, km, metadata = vpc.load_pca_kmeans('2014rhi_{n}comp'.format(n=n_comp))
    fields = metadata['fields']
    pn = vpc.dt2pn(dt0, dt1)
    data = vpc.prepare_data(pn, fields, metadata['hmax'])
    data_scaled = vpc.scale_data(data)
    
    fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis', n_extra_ax=0)
    classes = vpc.classify(data_scaled, km)
    
    for iax in range(len(axarr)-1):
        vpc.class_colors(classes, ax=axarr[iax])
