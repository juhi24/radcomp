#!/usr/bin/env python2
# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, insitu, classification, plotting, RESULTS_DIR
from j24 import ensure_dir

plt.ion()
plt.close('all')
np.random.seed(0)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'case_study'))
name = '140303'
n_eigens = 20
n_clusters = 20
reduced = True
#scheme = '2014rhi_{n}comp'.format(n=n_comp)
scheme = classification.scheme_name(basename='baecc+1415', n_eigens=n_eigens,
                                    n_clusters=n_clusters, reduced=reduced)

def plot_case(c, data_g):
    fig, axarr = c.plot(cmap='viridis', n_extra_ax=3)
    ax_r = axarr[-3]
    ax_rho = axarr[-2]
    ax_d = axarr[-1]
    plotting.plot_data(data_g['intensity'], ax_r)
    #data_g['intensity'].plot(drawstyle='steps', ax=ax_r) # bug?
    ax_r.set_ylabel('LWE, mm$\,$h$^{-1}$')
    ax_r.set_ylim([0, 6])
    plotting.plot_data(data_g['density'], ax_rho)
    ax_rho.set_ylabel('Density, kg$\,$m$^{-3}$')
    ax_rho.set_ylim([0, 600])
    plotting.plot_data(data_g['D_max'], ax_d, label='$D_{max}$')
    plotting.plot_data(data_g['D_0_gamma'], ax_d, label='$D_0$')
    ax_d.set_ylabel('mm')
    ax_d.set_ylim([0, 15])
    ax_d.legend()
    return fig, axarr

def plot_cases(cases, g, save=True, **kws):
    for name in cases.index.values:
        fig, axarr = plot_case(name, cases, g, **kws)
        fig.savefig(path.join(results_dir, name + '.png'))

cases = case.read_cases('analysis')
g = pd.read_pickle(insitu.TABLE_FILTERED_PKL)
data_g = g.loc[name]
c = cases.case[name]
c.load_classification(scheme)
fig, axarr = plot_case(c, data_g)
