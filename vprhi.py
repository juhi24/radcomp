#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import matplotlib as mpl
import copy
import collections
from os import path
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.cluster import KMeans
import learn

plt.ion()
plt.close('all')
np.random.seed(0)

datadir = '/home/jussitii/DATA/ToJussi'

def data_range(dt_start, dt_end):
    fnames = fname_range(dt_start, dt_end)
    pns = map(vprhimat2pn, fnames)
    return pd.concat(pns, axis=2).loc[:, :, dt_start:dt_end]

def fname_range(dt_start, dt_end):
    dt_range = pd.date_range(dt_start.date(), dt_end.date())
    return map(dt2path, dt_range)

def dt2path(dt, datadir='/home/jussitii/DATA/ToJussi'):
    return path.join(datadir, dt.strftime('%Y%m%d_IKA_VP_from_RHI.mat'))

def vprhimat2pn(datapath):
    data = scipy.io.loadmat(datapath)['VP_RHI']
    fields = list(data.dtype.fields)
    fields.remove('ObsTime')
    fields.remove('height')
    str2dt = lambda tstr: pd.datetime.strptime(tstr,'%Y-%m-%dT%H:%M:%S')
    t = list(map(str2dt, data['ObsTime'][0][0]))
    h = data['height'][0][0][0]
    data_dict = {}
    for field in fields:
        data_dict[field] = data[field][0][0].T
    return pd.Panel(data_dict, major_axis=h, minor_axis=t)

def plotpn(pn, fields=None, **kws):
    if fields is None:
        fields = pn.items
    vmin = {'ZH': -15, 'ZDR': -1, 'RHO': 0, 'KDP': 0}
    vmax = {'ZH': 30, 'ZDR': 4, 'RHO': 1, 'KDP': 0.26}
    label = {'ZH': 'dBZ', 'ZDR': 'dB', 'KDP': 'deg/km'}
    fig, axarr = plt.subplots(len(fields), sharex=True, sharey=True)
    if not isinstance(axarr, collections.Iterable):
        axarr = [axarr]
    def m2km(m, pos):
        return '{:.0f}'.format(m*1e-3)
    for i, field in enumerate(fields):
        ax = axarr[i]
        im = ax.pcolormesh(pn[field].columns, pn[field].index, 
                      np.ma.masked_invalid(pn[field].values), cmap='gist_ncar',
                      vmin=vmin[field], vmax=vmax[field], label=field, **kws)
        #fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
        ax.set_ylim(0,10000)
        ax.set_ylabel('Height, km')
        fig.colorbar(im, ax=ax, label=label[field])
    ax.set_xlabel('Time, UTC')
    axarr[0].set_title(str(pn[field].columns[0].date()))
    fig.tight_layout()
    return fig, axarr

def scale_data(pn):
    scaling_limits = {'ZH': (-10, 30), 'ZDR': (-1, 4), 'KDP': (0, 0.25)}
    scaled = copy.deepcopy(pn)
    for field, data in scaled.iteritems():
        data -= scaling_limits[field][0]
        data *= 1.0/scaling_limits[field][1]
        scaled[field] = data
    return scaled

def fillna(pn):
    nan_replacement = {'ZH': -10, 'ZDR': 0, 'KDP': 0}
    for field in list(pn.items):
        pn[field].fillna(nan_replacement[field], inplace=True)
    return pn

dt0 = pd.datetime(2014, 2, 21, 15, 30)
dt1 = pd.datetime(2014, 2, 22, 5, 30)
pn = data_range(dt0, dt1)
fig, axarr = plotpn(pn, fields=['ZH', 'ZDR', 'KDP'])

fields = ['ZH', 'ZDR', 'KDP']
plot_components = True
hmax = 10000
n_eigens = 4
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
data = fillna(pn[fields, 0:hmax, :]).transpose(0,2,1)
#data_scaled = copy.deepcopy(data)
#data_scaled['ZH'] = preprocessing.scale(data['ZH'], axis=1)
data_uniscaled = scale_data(data)
plotpn(data.transpose(0,2,1))
#plotpn(data_scaled.transpose(0,2,1)*5)
plotpn(data_uniscaled.transpose(0,2,1)*5)
data_df = learn.pn2df(data)
pca.fit(data_df)
if plot_components:
    learn.plot_pca_components(pca, data)

learn.pca_stats(pca)
km = KMeans(init=pca.components_, n_clusters=n_eigens, n_init=1)
km.fit(data_df)
classes = km.labels_

for eigen in range(n_eigens):   
    i_classes = np.where(classes==eigen)[0]
    pn_class = data.iloc[:, i_classes, :]
    learn.plot_class(pn_class, ylim=(-10, 40))
