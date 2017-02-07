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
from sklearn.cluster import KMeans
from scipy import signal
from scipy.ndimage.filters import median_filter
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

def plotpn(pn, fields=None, scaled=False, cmap='gist_ncar', **kws):
    if fields is None:
        fields = pn.items
    vmins = {'ZH': -15, 'ZDR': -1, 'zdr': -1, 'RHO': 0, 'KDP': 0, 'kdp': 0, 'DP': 0, 'phidp': 0}
    vmaxs = {'ZH': 30, 'ZDR': 4, 'zdr': 4, 'RHO': 1, 'KDP': 0.26, 'kdp': 0.26, 'DP': 360, 'phidp': 360}
    labels = {'ZH': 'dBZ', 'ZDR': 'dB', 'zdr': 'dB', 'KDP': 'deg/km', 'kdp': 'deg/km', 'DP': 'deg'}
    fig, axarr = plt.subplots(len(fields), sharex=True, sharey=True)
    if not isinstance(axarr, collections.Iterable):
        axarr = [axarr]
    def m2km(m, pos):
        return '{:.0f}'.format(m*1e-3)
    for i, field in enumerate(fields):
        ax = axarr[i]
        if scaled:
            scalekws = {'vmin': 0, 'vmax': 1}
            label = 'scaled'
        elif field in labels:
            scalekws = {'vmin': vmins[field], 'vmax': vmaxs[field]}
            label = labels[field]
        else:
            scalekws = {}
            label = field
        im = ax.pcolormesh(pn[field].columns, pn[field].index, 
                      np.ma.masked_invalid(pn[field].values), cmap=cmap,
                      **scalekws, label=field, **kws)
        #fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
        ax.set_ylim(0,11000)
        ax.set_ylabel('Height, km')
        fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel('Time, UTC')
    axarr[0].set_title(str(pn[field].columns[0].date()))
    fig.tight_layout()
    return fig, axarr

def scale_data(pn):
    scaling_limits = {'ZH': (-10, 30), 'ZDR': (0, 3), 'KDP': (0, 0.5), 
                      'kdp': (0, 0.15)}
    scaled = copy.deepcopy(pn)
    for field, data in scaled.iteritems():
        data -= scaling_limits[field][0]
        data *= 1.0/scaling_limits[field][1]
        scaled[field] = data
    return scaled

def fillna(pn):
    nan_replacement = {'ZH': -10, 'ZDR': 0, 'zdr': 0, 'KDP': 0, 'kdp': 0}
    for field in list(pn.items):
        pn[field].fillna(nan_replacement[field], inplace=True)
    return pn

def kdp2phidp(kdp, dr_km):
    kdp_filled = kdp.fillna(0)
    return 2*kdp_filled.cumsum().multiply(dr_km, axis=0)

def prepare_pn(pn, kdpmax=2, fltr_size=(20,2)):
    dr = pd.Series(pn.major_axis.values, index=pn.major_axis).diff().bfill()
    dr_km = dr/1000
    pn_new = copy.deepcopy(pn)
    pn_new['KDP_orig'] = pn_new['KDP'].copy()
    pn_new['KDP'][pn_new['KDP']<0] = np.nan
    pn_new['phidp'] = kdp2phidp(pn_new['KDP'], dr_km)
    nullmask = pn_new['KDP_orig'].isnull()
    kdp = pn_new['KDP'].fillna(0)
    kdp[kdp>kdpmax] = 0
    kdp[kdp<0] = 0
    kdp_filtered = median_filter(kdp, size=fltr_size)
    kdp_filtered[nullmask] = np.nan
    pn_new['kdp'] = kdp_filtered
    return pn_new

def prepare_data(pn, fields=['ZH', 'ZDR', 'kdp'], hmax=10e3, kdpmax=None):
    data = pn[fields, 0:hmax, :].transpose(0,2,1)
    if kdpmax is not None:
        data['KDP'][data['KDP']>kdpmax] = np.nan
    return fillna(data)

def class_colors(classes, ymin=-0.2, ymax=0, ax=None, cmap='Vega10', alpha=1, **kws):
    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    t0 = classes.index[0]
    for t1, icolor in classes.iteritems():
        if t1==t0:
            continue
        ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, facecolor=cm.colors[icolor], 
                   alpha=alpha, clip_on=False, **kws)
        t0 = t1

def plot_classes(data, classes, n_eigens):
    for eigen in range(n_eigens):
        i_classes = np.where(classes==eigen)[0]
        pn_class = data.iloc[:, i_classes, :]
        learn.plot_class(pn_class, ylim=(-1, 2))

def reject_outliers(df, m=2):
    d = df.subtract(df.median(axis=1), axis=0).abs()
    mdev = d.median(axis=1)
    s = d.divide(mdev, axis=0).replace(np.inf, np.nan).fillna(0)
    return df[s<m].copy()

def filter_test(sig, n, wn, **kws):
    #fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True)
    fig, ax1 = plt.subplots()
    ax2 = ax1
    b, a = signal.butter(n, wn, **kws)
    x = sig.index.values
    y = signal.filtfilt(b, a, sig)
    sig.plot(ax=ax1)
    ax2.plot(x,y)

def rolling_filter(df, window=5, stdlim=0.1, fill_value=0, **kws):
    r = df.rolling(window=window, center=True)

dt0 = pd.datetime(2014, 2, 21, 15, 30)
dt1 = pd.datetime(2014, 2, 22, 15, 30)
pn_raw = data_range(dt0, dt1)

pn = prepare_pn(pn_raw)
fields = ['ZH', 'ZDR', 'kdp']
fig, axarr = plotpn(pn, fields=fields)

plot_components = True
hmax = 10000
n_eigens = 10
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
data = prepare_data(pn, fields, hmax)
data_scaled = scale_data(data)
plotpn(data.transpose(0,2,1))
plotpn(data_scaled.transpose(0,2,1), scaled=True)
data_df = learn.pn2df(data_scaled)
pca.fit(data_df)
if plot_components:
    learn.plot_pca_components(pca, data_scaled)

learn.pca_stats(pca)
km = KMeans(init=pca.components_, n_clusters=n_eigens, n_init=1)
km.fit(data_df)
classes = pd.Series(data=km.labels_, index=pn.minor_axis)

for iax in [0,1]:
    class_colors(classes, ax=axarr[iax])

data['zdr']=median_filter(data['ZDR'], size=(1,6))
plotpn(data[['ZDR', 'zdr']].transpose(0,2,1), cmap='viridis')


#plot_classes(data_scaled, classes, n_eigens)
def hohoo():
    kdp1=pn['KDP'].fillna(0).iloc[:,56].copy()
    kdp1sort = kdp1.sort_values(ascending=False)
    plt.figure()
    plt.plot(kdp1sort.values)

def medfiltest(size=(20,2), kdpmax=2):
    kdp = pn[['KDP']].fillna(0)
    kdp['KDP'][kdp['KDP']>kdpmax] = 0
    kdp['kdp'] = median_filter(kdp['KDP'], size=size)
    plotpn(kdp, cmap='viridis')

#pn['kdpc']=wradlib.dp.kdp_from_phidp_convolution(pn['DP'].values.T).T
#phidp, kdp = wradlib.dp.process_raw_phidp_vulpiani(pn['DP'].T.values, 0.05, copy=True)
#pn['phidp'] = phidp.T
#plotpn(pn[['DP', 'phidp']])
#plotpn(pn[['KDP', 'kdpc']])
