# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
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
n_eigens = 25
n_clusters = 20
use_temperature = True

cases = case.read_cases('training_baecc')
basename = 'baecc'

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced)
c = case.Case.by_combining(cases, class_scheme=scheme)
trainkws = {}
if reduced:
    trainkws['n_clusters'] = n_clusters
    trainkws['use_temperature'] = use_temperature
c.train(**trainkws)
if use_temperature:
    basename += '_t'
name = classification.scheme_name(basename=basename, n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
scheme.save(name)

