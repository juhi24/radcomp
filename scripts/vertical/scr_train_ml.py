# coding: utf-8
"""
A script for using the VP classification training method
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


cases = case.read_cases('melting')
cases = cases[cases.ml_ok.astype(bool)]
basename = 'mlt'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 20
n_clusters = 15
reduced = True
use_temperature = False
radar_weight_factors = dict()

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            reduced=reduced,
                            radar_weight_factors=radar_weight_factors,
                            basename=basename, n_clusters=n_clusters)
c = case.Case.by_combining(cases, class_scheme=scheme, has_ml=True)
c.train()
scheme.save()
# Load classification and plot centroids
name = c.class_scheme.name()
print(name)
c.load_classification(name)
order = c.clus_centroids()[0].ZH.iloc[0]
c.plot_cluster_centroids(cmap='viridis', colorful_bars='blue', sortby=order)

c.scatter_class_pca(plot3d=True)


