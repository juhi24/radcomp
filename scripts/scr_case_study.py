#!/usr/bin/env python2
# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, RESULTS_DIR
from j24 import ensure_dir

plt.ion()
plt.close('all')
np.random.seed(0)

results_dir = ensure_dir(path.join(RESULTS_DIR, 'case_study'))
cases = case.read_cases('analysis')
home = path.expanduser('~')

def plot_data(data, ax, **kws):
    return ax.plot(data.index, data.values, drawstyle='steps', **kws)

def fltr_long_period(data_in, minutes=30):
    data = data_in.copy()
    selection = data.tdelta>pd.tseries.offsets.Minute(minutes).delta
    data.loc[selection,['density','D_max','D_0_gamma', 'N_w']] = np.nan
    return data

def plot_case(name, n_comp=20):
    scheme = '2014rhi_{n}comp'.format(n=n_comp)
    c = cases.case[name]
    c.load_classification(scheme)
    fig, axarr = c.plot(cmap='viridis', n_extra_ax=3)
    
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

plot_case('mar3')
