#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
import radx
from os import path

plt.ioff()
plt.close('all')
np.random.seed(0)

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

store = pd.HDFStore(storefp)
pn = store['s2016'].sort_index(1, ascending=False)
rh = pn.iloc[-1].RELH
rhi = interp_akima(rh)

n_eigens = 8
var = 'RELH'
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
dat = interp_akima(pn.loc[:,:,var]).loc[900:100].T
n_samples, n_features = dat.shape
pca.fit(dat)
for i in range(pca.components_.shape[0]):
    plt.figure()
    plt.plot(pca.components_[i])
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
