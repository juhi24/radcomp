#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.cluster import KMeans
import radx
from os import path
from matplotlib import gridspec

import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
fmr = importr('FactoMineR')

plt.ion()
plt.close('all')
np.random.seed(0)
plot_components = True
plot_individual = False

storefp = '/home/jussitii/DATA/soundings.hdf'
resultspath = '/home/jussitii/results/radcomp/soundings'

def sounding_url(datetime):
    year = datetime.year
    mon = str(datetime.month).zfill(2)
    day = str(datetime.day).zfill(2)
    hour = str(datetime.hour).zfill(2)
    urlformat = 'http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}{hour}&TO={day}{hour}&STNM=02963'
    return urlformat.format(year=year, month=mon, day=day, hour=hour)

def read_sounding(datetime):
    skiprows = (0,1,2,3,4,5,6,8,9)
    url = sounding_url(datetime)
    data = pd.read_table(url, delim_whitespace=True, index_col=0, skiprows=skiprows).dropna()
    data = data.drop(data.tail(1).index).astype(np.float)
    data.index = data.index.astype(np.float)
    return data

def create_pn():
    dt_start = pd.datetime(2016, 1, 1, 00)
    dt_end = pd.datetime(2016, 5, 1, 00)
    dt_range = pd.date_range(dt_start, dt_end)
    d = {}
    for dt in dt_range:
        print(str(dt))
        d[dt] = read_sounding(dt)
    pn = pd.Panel(d)
    return pn

def create_hdf(filepath='/home/jussitii/DATA/soundings.hdf', key='s2016'):
    with pd.HDFStore(filepath) as store:
        store[key] = create_pn()

def comp_interp_methods(rh):
    methods = ('linear', 'index', 'values', 'nearest', 'zero','slinear', 'quadratic', 'cubic', 'barycentric', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima')
    for method in methods:
        interp = rh.interpolate(method=method)
        plt.figure()
        interp.plot(label=method)
        plt.legend()

def reindex(data, force_asc=False):
    i=np.array(range(int(data.index.min()*10),int(data.index.max()*10+1)))/10.
    ii = i[::-1]
    if force_asc:
        return data.reindex(i)
    return data.reindex(ii)

def interp_akima(data):
    data = reindex(data, force_asc=True)
    intrp = data.interpolate(method='akima')
    return intrp.sort_index(ascending=False)

def sq_subplots(n_axes, use_gs=True, **kws):
    n_rows_cols = int(np.ceil(np.sqrt(n_axes)))
    if use_gs:
        return ncols_subplots(n_axes, n_cols=n_rows_cols, **kws)
    return plt.subplot(n_rows_cols, n_rows_cols, **kws)

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

def scale_t(df):
    arr = preprocessing.scale(df.T).T
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def pn2df(pn):
    pass

store = pd.HDFStore(storefp)
pn = store['s2016'].sort_index(1, ascending=False) # s2016
rh = pn.iloc[-1].RELH
rhi = interp_akima(rh)

n_eigens = 8
fields = ('RELH', 'TEMP', 'DWPT')
var = 'RELH'
rawdat = pn.loc[:, :, fields]
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
#pca = decomposition.PCA(n_components=0.95, svd_solver='full', whiten=True)
dat_dict = {}
for field in fields:
     dat_dict[field] = scale_t(interp_akima(rawdat.loc[:, :, field]).loc[900:100].T)
dat_pn = pd.Panel(dat_dict)
n_fields, n_samples, n_features = dat_pn.shape
dat = dat_pn[var]
pca.fit(dat)
if plot_components:
    fig_comps, axarr_comps = sq_subplots(n_eigens, sharex=True)
    for i in range(n_eigens):
        ax = axarr_comps.flatten()[i]
        ax.plot(pca.components_[i])
with plt.style.context('fivethirtyeight'):
    plt.figure();
    plt.title('Explained variance ratio over component');
    plt.plot(pca.explained_variance_ratio_);
with plt.style.context('fivethirtyeight'):
    plt.figure();
    plt.title('Cumulative explained variance over eigen' + var);
    plt.plot(pca.explained_variance_ratio_.cumsum());
print('PCA captures {:.2f} percent of the variance in the dataset.'.format(pca.explained_variance_ratio_.sum() * 100))

km = KMeans(init=pca.components_, n_clusters=n_eigens, n_init=1)
km.fit(dat)
classes = km.labels_

for eigen in range(n_eigens):
    i_classes = np.where(classes==eigen)[0]
    fig_class, axarr_class = ncols_subplots(i_classes.size, n_cols=5)
    for i, i_class in enumerate(i_classes):
        ax = axarr_class.flatten()[i]
        data = dat.iloc[i_class]
        data.plot(ax=ax)
        ax.set_xlabel('')
    #ax.set_xlabel('Pressure (hPa)')
    #ax.set_ylabel('RH')
    fname = str(i) + '.eps'
    savepath = radx.ensure_path(path.join(resultspath, var))
    fig_class.savefig(path.join(savepath, fname))

if plot_individual:
    for i in range(n_samples):
        data = dat.iloc[i]
        savepath = radx.ensure_path(path.join(resultspath, var, str(classes[i])))
        fname = data.name.strftime('%Y%m%d-%HZ.eps')
        fig = plt.figure()
        ax = data.plot()
        ax.set_xlabel('Pressure (hPa)')
        ax.set_ylabel('RH')
        fig.savefig(path.join(savepath, fname))
        plt.close(fig)