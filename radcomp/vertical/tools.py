#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Vertical profile classification
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from radcomp import learn

HOME = path.expanduser('~')
USER_DIR = path.join(HOME, '.radcomp')
RESULTS_DIR = path.join(HOME, 'results', 'radcomp', 'vertical')
META_SUFFIX = '_metadata'
NAN_REPLACEMENT = {'ZH': -10, 'ZDR': 0, 'KDP': 0}

def case_id_fmt(t):
    return t.strftime('%b%-d').lower()

def read_cases(name):
    filepath = path.join(USER_DIR, 'cases', name + '.csv')
    dts = pd.read_csv(filepath, parse_dates=['t_start', 't_end'])
    dts.index = dts['t_start'].apply(case_id_fmt)
    dts.index.name = 'id'
    return dts

def mean_delta(t):
    dt = t[-1]-t[0]
    return dt/(len(t)-1)

def m2km(m, pos):
    '''formatting m in km'''
    return '{:.0f}'.format(m*1e-3)

def plotpn(pn, fields=None, scaled=False, cmap='gist_ncar', n_extra_ax=0, **kws):
    if fields is None:
        fields = pn.items
    vmins = {'ZH': -15, 'ZDR': -1, 'RHO': 0, 'KDP': 0, 'DP': 0, 'PHIDP': 0}
    vmaxs = {'ZH': 30, 'ZDR': 4, 'RHO': 1, 'KDP': 0.26, 'DP': 360, 'PHIDP': 360}
    labels = {'ZH': 'dBZ', 'ZDR': 'dB', 'KDP': 'deg/km', 'DP': 'deg', 'PHIDP': 'deg'}
    n_rows = len(fields) + n_extra_ax
    fig = plt.figure(figsize=(8,3+1.1*n_rows))
    gs = mpl.gridspec.GridSpec(n_rows, 2, width_ratios=(35, 1), wspace=0.02,
                           top=1-0.22/n_rows, bottom=0.35/n_rows, left=0.1, right=0.905)
    axarr = []
    for i, field in enumerate(fields):
        subplot_kws = {}
        if i>0:
            subplot_kws['sharex'] = axarr[0]
        ax = fig.add_subplot(gs[i, 0], **subplot_kws)
        ax_cb = fig.add_subplot(gs[i, 1])
        axarr.append(ax)
        fieldup = field.upper()
        if scaled:
            scalekws = {'vmin': 0, 'vmax': 1}
            label = 'scaled'
        elif fieldup in labels:
            scalekws = {'vmin': vmins[fieldup], 'vmax': vmaxs[fieldup]}
            label = labels[fieldup]
        else:
            scalekws = {}
            label = field
        t = pn[field].columns
        t_shifted = t + mean_delta(t)/2
        im = ax.pcolormesh(t_shifted, pn[field].index, 
                      np.ma.masked_invalid(pn[field].values), cmap=cmap,
                      **scalekws, label=field, **kws)
        #fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
        ax.set_ylim(0,11000)
        ax.set_ylabel('Height, km')
        #fig.colorbar(im, ax=ax, label=label)
        fig.colorbar(im, cax=ax_cb, label=label)
    for j in range(n_extra_ax):
        ax = fig.add_subplot(gs[i+1+j, 0], sharex=axarr[0])
        axarr.append(ax)
    axarr[-1].set_xlabel('Time, UTC')
    axarr[0].set_title(str(pn[field].columns[0].date()))
    for ax in axarr[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    return fig, axarr

def scale_data(pn):
    scaling_limits = {'ZH': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3), 'KDP': (0, 0.5), 
                      'kdp': (0, 0.15)}
    scaled = pn.copy()
    for field, data in scaled.iteritems():
        data -= scaling_limits[field][0]
        data *= 1.0/scaling_limits[field][1]
        scaled[field] = data
    return scaled

def fillna(dat, field=''):
    data = dat.copy()
    if isinstance(data, pd.Panel):
        for field in list(data.items):
            data[field].fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    elif isinstance(data, pd.DataFrame):
        data.fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    return data

def prepare_data(pn, fields=['ZH', 'ZDR', 'kdp'], hmax=10e3, kdpmax=None):
    data = pn[fields, 0:hmax, :].transpose(0,2,1)
    if kdpmax is not None:
        data['KDP'][data['KDP']>kdpmax] = np.nan
    return fillna(data)

def class_colors(classes, ymin=-0.2, ymax=0, ax=None, cmap='Vega20', alpha=1, **kws):
    t = classes.index
    dt = mean_delta(t)*1.5
    clss = classes.shift(freq=dt).dropna().astype(int)
    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    t0 = clss.index[0]
    for t1, icolor in clss.iteritems():
        if t1<=t0:
            continue
        ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, facecolor=cm.colors[icolor], 
                   alpha=alpha, clip_on=False, **kws)
        t0 = t1

def plot_classes(data, classes):
    figs = []
    axarrs = []
    n_classes = classes.max()+1
    for eigen in range(n_classes):
        i_classes = np.where(classes==eigen)[0]
        if len(i_classes)==0:
            continue
        pn_class = data.iloc[:, i_classes, :]
        fig, axarr = learn.plot_class(pn_class, ylim=(-1, 2))
        axarr[0].legend().set_visible(True)
        figs.append(fig)
        axarrs.append(axarr)
        for ax in axarr:
            if ax.xaxis.get_ticklabels()[0].get_visible():
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
    return figs, axarrs

def model_path(name):
    return path.join(RESULTS_DIR, 'models', name + '.pkl')

def save_model(model, name):
    savepath = model_path(name)
    joblib.dump(model, savepath)
    return savepath

def save_data(data, name):
    joblib.dump(data, model_path(name + '_data'))
    hmax = np.ceil(data.minor_axis.max())
    fields = list(data.items)
    metadata = dict(hmax=hmax, fields=fields)
    joblib.dump(metadata, model_path(name + META_SUFFIX))

def load_model(name):
    loadpath = model_path(name)
    model = joblib.load(loadpath)
    return model

def save_pca_kmeans(pca, kmeans, data, name):
    save_model(pca, name + '_pca')
    save_model(kmeans, name + '_kmeans')
    save_data(data, name)

def load_pca_kmeans(name):
    '''return pca, km, metadata'''
    pca = load_model(name + '_pca')
    km = load_model(name + '_kmeans')
    metadata = joblib.load(model_path(name + META_SUFFIX))
    return pca, km, metadata

def train(data_scaled, n_eigens, quiet=False, **kws):
    data_df = learn.pn2df(data_scaled)
    pca = pca_fit(data_df, n_components=n_eigens)
    if not quiet:
        learn.pca_stats(pca)
    km = kmeans(data_df, pca)
    return pca, km

def pca_fit(data_df, whiten=True, **kws):
    pca = decomposition.PCA(whiten=whiten, **kws)
    pca.fit(data_df)
    return pca

def kmeans(data_df, pca):
    km = KMeans(init=pca.components_, n_clusters=pca.n_components, n_init=1)
    km.fit(data_df)
    return km

def classify(data_scaled, km):
    data_df = learn.pn2df(data_scaled)
    return pd.Series(data=km.predict(data_df), index=data_scaled.major_axis)

