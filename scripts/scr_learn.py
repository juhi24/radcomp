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
from j24.stats import pca_stats

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
scaling = False
n_eigens = 10

storefp = '/home/jussitii/DATA/soundings.hdf'
resultspath = '/home/jussitii/results/radcomp/soundings'

def prepare_pn(pn, fields=['TEMP', 'DWPT']):
    pn0 = pn[:,:,pn.TEMP.max()<0]
    dat_dict = {}
    for field in fields:
        dd = pn0[field]
        dd = sounding.interp_akima(dd)
        dd = dd.loc[950:100].T
        dat_dict[field] = dd
    return pd.Panel(dat_dict)

store = pd.HDFStore(storefp)
pn = store['w1516'].sort_index(1, ascending=False) # s2016, s2016_12h

#tmax = pd.concat({'PRES': pn.TEMP.idxmax(), 'TEMP': pn.TEMP.max()}, axis=1)
pca = decomposition.PCA(n_components=n_eigens, whiten=True)
#pca = decomposition.PCA(n_components=0.95, svd_solver='full', whiten=True)

fields = ['TEMP', 'DWPT']
dat_pn = prepare_pn(pn, fields=fields)
dat_df = learn.pn2df(dat_pn).dropna()
if scaling:
    dat_df = preprocessing.scale(dat_df)
pca.fit(dat_df)
fig_comps, axarr_comps = learn.plot_pca_components(pca, dat_pn)
axarr_comps[0].set_xticks((100,500,900))
axarr_comps[0].invert_xaxis() # invert shared xaxis only once
pca_stats(pca)

km = KMeans(init=pca.components_, n_clusters=n_eigens, n_init=1)
km.fit(dat_df)
classes = km.labels_

for eigen in range(n_eigens):
    i_classes = np.where(classes==eigen)[0]
    dat_pn_class = dat_pn.iloc[:, i_classes, :]
    fig_class, axarr_class = learn.plot_class(dat_pn_class,
                                                xticks=(100,500,900), 
                                                yticks=range(-80, 40, 20),
                                                ylim=(-90, 10))
    if not debug:
        fname = str(eigen) + '.eps'
        savepath = radx.ensure_path(path.join(resultspath, 'sounding'))
        fig_class.savefig(path.join(savepath, fname))
