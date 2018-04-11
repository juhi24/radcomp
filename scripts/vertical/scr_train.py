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
np.random.seed(1)

plot = False

cases = case.read_cases('14-16by_hand')
basename = '14-16'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced, t_weight_factor=t_weight_factor,
                            radar_weight_factors=radar_weight_factors,
                            basename=basename)
c = case.Case.by_combining(cases, class_scheme=scheme)
c.train()
scheme.save()
# Load classification and plot centroids
name = c.class_scheme.name()
print(name)
c.load_classification(name)
c.plot_cluster_centroids(cmap='viridis', colorful_bars='blue')

