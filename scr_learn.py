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
import sounding
import learn

#import rpy2.robjects as ro
#from rpy2.robjects import r
#from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri
#pandas2ri.activate()
#fmr = importr('FactoMineR')

plt.ion()
plt.close('all')
np.random.seed(0)
debug = True
plot_components = True
plot_individual = False
scaling = False
n_eigens = 10

storefp = '/home/jussitii/DATA/soundings.hdf'
resultspath = '/home/jussitii/results/radcomp/soundings'

def pn2df(pn, axis=1, **kws):
    return pd.concat([pn[item] for item in pn.items], axis=axis, **kws)

store = pd.HDFStore(storefp)
pn = store['s2016_12h'].sort_index(1, ascending=False) # s2016, s2016_12h

#tmax = pd.concat({'PRES': pn.TEMP.idxmax(), 'TEMP': pn.TEMP.max()}, axis=1)
pn0 = pn[:,:,pn.TEMP.max()<0]
fields = ['TEMP', 'DWPT']
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
#pca = decomposition.PCA(n_components=0.95, svd_solver='full', whiten=True)
dat_dict = {}
for field in fields:
    dd = pn0[field]
    dd = sounding.interp_akima(dd)
    dd = dd.loc[950:100].T
    dat_dict[field] = dd
dat_pn = pd.Panel(dat_dict)
dat_df = pn2df(dat_pn).dropna()
if scaling:
    dat_df = preprocessing.scale(dat_df)
n_samples, n_features = dat_df.shape
x_len = dat_pn.minor_axis.size
pca.fit(dat_df)
if plot_components:
    fig_comps, axarr_comps = learn.sq_subplots(n_eigens, sharex=True)
    axarr_comps_flat = axarr_comps.flatten()
    for i in range(n_eigens):
        ax = axarr_comps_flat[i]
        comps = pca.components_[i].reshape((len(fields), x_len))
        for comp in comps:
            x = list(dat_pn.minor_axis)
            ax.plot(x, comp)
        ax.set_xticks((100,500,900))
    ax.invert_xaxis() # invert shared xaxis only once
learn.pca_stats(pca)

km = KMeans(init=pca.components_, n_clusters=n_eigens, n_init=1)
km.fit(dat_df)
classes = km.labels_

for eigen in range(n_eigens):
    i_classes = np.where(classes==eigen)[0]
    fig_class, axarr_class = learn.plot_classes(dat_df, i_classes,
                                                xticks=(100,500,900), 
                                                yticks=range(-80, 40, 20),
                                                ylim=(-90, 10))
    if not debug:
        fname = str(i) + '.eps'
        savepath = radx.ensure_path(path.join(resultspath, 'sounding'))
        fig_class.savefig(path.join(savepath, fname))

if plot_individual:
    for i in range(n_samples):
        data = dat_pn.iloc[:,i,:]
        savepath = radx.ensure_path(path.join(resultspath, 'sounding', str(classes[i])))
        fname = data.name.strftime('%Y%m%d-%HZ.eps')
        fig = plt.figure()
        ax = data.plot()
        ax.set_xlabel('Pressure (hPa)')
        ax.set_ylabel('RH')
        fig.savefig(path.join(savepath, fname))
        plt.close(fig)