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

#dt0 = pd.datetime(2014, 2, 21, 12, 30)
#dt1 = pd.datetime(2014, 2, 22, 15, 30)
n_comp = 20

def load_case(dt0, dt1):
    pca, km, metadata = vpc.load_pca_kmeans('2014rhi_{n}comp'.format(n=n_comp))
    fields = metadata['fields']
    pn = vpc.dt2pn(dt0, dt1)
    data = vpc.prepare_data(pn, fields, metadata['hmax'])
    data_scaled = vpc.scale_data(data)
    classes = vpc.classify(data_scaled, km)
    return {'data': data, 'data_scaled': data_scaled, 'classes': classes,
            'metadata': metadata, 'kmeans': km, 'pca': pca, 'pn': pn}

def plot_case(case):
    fig, axarr = vpc.plotpn(pn, fields=fields, cmap='viridis', n_extra_ax=0)
    for iax in range(len(axarr)-1):
        vpc.class_colors(classes, ax=axarr[iax])
    return fig, axarr


cases = vpc.read_cases('analysis')
t_case = cases.loc['mar3']
dt0 = t_case.t_start
dt1 = t_case.t_end

plot_case(dt0, dt1)
vpc.plot_classes(data_scaled, classes)