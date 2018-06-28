# coding: utf-8
"""
A script for using the VP classification training method
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification, RESULTS_DIR
from j24 import ensure_join


plt.ion()
plt.close('all')
np.random.seed(1)

savefig = False

cases = multicase.read_cases('14-16by_hand')
basename = '14-16'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced, t_weight_factor=t_weight_factor,
                            radar_weight_factors=radar_weight_factors,
                            basename=basename, n_clusters=n_clusters,
                            use_temperature=use_temperature)
c = multicase.MultiCase.by_combining(cases, class_scheme=scheme)
trainkws = {}
#if reduced:
    #trainkws['n_clusters'] = n_clusters
    #trainkws['use_temperature'] = use_temperature
c.train(**trainkws)
scheme.save()
# Load classification and plot centroids
name = c.class_scheme.name()
print(name)
c.load_classification(name)
fig, axarr, i = c.plot_cluster_centroids(colorful_bars='blue')
if savefig:
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name)
    savefile = path.join(savedir, 'centroids.png')
    fig.savefig(savefile, bbox_inches='tight')
