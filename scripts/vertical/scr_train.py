# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp import RESULTS_DIR
from radcomp.vertical import case, classification

plt.ion()
plt.close('all')
np.random.seed(0)

plot = False
reduced = True

params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 20
n_clusters = 20

cases = case.read_cases('training')

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

pns = [c.data for i,c in cases.case.iteritems()]
data = pd.concat(pns, axis=2)
scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced)
c = case.Case(data=data, class_scheme=scheme)
trainkws = {}
if reduced:
    trainkws['n_clusters'] = n_clusters
c.train(**trainkws)
name = classification.scheme_name(basename='baecc+1415', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
scheme.save(name)

