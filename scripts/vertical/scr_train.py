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

cases = case.read_cases('14-16by_hand')
basename = '14-16'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 24
n_clusters = 24
reduced = True
use_temperature = True
t_weight_factor = 0.3

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced, t_weight_factor=t_weight_factor)
c = case.Case.by_combining(cases, class_scheme=scheme)
trainkws = {}
if reduced:
    trainkws['n_clusters'] = n_clusters
    trainkws['use_temperature'] = use_temperature
c.train(**trainkws)
name = classification.scheme_name(basename=basename, n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor)
scheme.save(name)
# Load classification and plot centroids
c.load_classification(name)
c.plot_cluster_centroids(cmap='viridis')

