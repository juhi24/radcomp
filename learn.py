#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

def ncols_subplots(n_axes, n_cols=3, sharex=False, sharey=False):
    n_rows = int(np.ceil(float(n_axes)/n_cols))
    gs = gridspec.GridSpec(n_rows, n_cols)
    fig = plt.figure(figsize=(n_cols*2.5+0.5, n_rows*2.5+0.5))
    axarr = []
    for i in range(n_axes):
        subplot_kws = {}
        if sharex and i>0:
            subplot_kws['sharex'] = axarr[0]
        if sharey and i>0:
            subplot_kws['sharey'] = axarr[0]
        axarr.append(fig.add_subplot(gs[i], **subplot_kws))
    return fig, np.array(axarr)

def sq_subplots(n_axes, use_gs=True, **kws):
    n_rows_cols = int(np.ceil(np.sqrt(n_axes)))
    if use_gs:
        return ncols_subplots(n_axes, n_cols=n_rows_cols, **kws)
    return plt.subplot(n_rows_cols, n_rows_cols, **kws)

def plot_class(pn_class, n_cols=5, **kws):
    fig, axarr = ncols_subplots(pn_class.shape[1], n_cols=n_cols)
    for i, key in enumerate(pn_class.major_axis):
        rows = pn_class.major_xs(key)
        ax = axarr.flatten()[i]
        rows.plot(ax=ax, **kws)
        ax.set_xlabel('')
        ax.legend().set_visible(False)
    return fig, axarr

def pca_stats(pca):
    with plt.style.context('fivethirtyeight'):
        plt.figure();
        plt.title('Explained variance ratio over component');
        plt.plot(pca.explained_variance_ratio_);
    with plt.style.context('fivethirtyeight'):
        plt.figure();
        plt.title('Cumulative explained variance over eigensounding');
        plt.plot(pca.explained_variance_ratio_.cumsum());
    print('PCA captures {:.2f}% of the variance in the dataset.'.format(pca.explained_variance_ratio_.sum() * 100))

def plot_pca_components(pca, pn):
    fig, axarr = sq_subplots(pca.n_components, sharex=True)
    axarr_flat = axarr.flatten()
    for i in range(pca.n_components):
        ax = axarr_flat[i]
        comps = pca.components_[i].reshape((pn.items.size, pn.minor_axis.size))
        for comp in comps:
            x = list(pn.minor_axis)
            ax.plot(x, comp)
    return fig, axarr

def pn2df(pn, axis=1, **kws):
    return pd.concat([pn[item] for item in pn.items], axis=axis, **kws)
