# coding: utf-8
"""
A script for using the VP classification training method
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification, RESULTS_DIR
from j24 import ensure_join


save_plots = True
train = True

cases = multicase.read_cases('melting')
cases = cases[cases.ml_ok.fillna(0).astype(bool)]
basename = 'mlt2'
params = ['ZH', 'zdr', 'kdp']
hlimits = (290, 10e3)
n_eigens = 16
n_clusters = 17
reduced = True
use_temperature = False
radar_weight_factors = dict()


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    np.random.seed(1)
    scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                                reduced=reduced,
                                radar_weight_factors=radar_weight_factors,
                                basename=basename, n_clusters=n_clusters)
    c = multicase.MultiCase.by_combining(cases, class_scheme=scheme, has_ml=True)
    if train:
        c.train()
        scheme.save()
    # Load classification and plot centroids
    c.load_classification()
    fig_cc, axarr_cc, i = c.plot_cluster_centroids(#cmap='viridis',
                                                   colorful_bars='blue',
                                                   sortby=None)

    c.scatter_class_pca(plot3d=True)
    fig_s, ax_s = plt.subplots()
    c.plot_silhouette(ax=ax_s)
    if save_plots:
        savedir = ensure_join(RESULTS_DIR, 'classes_summary', c.class_scheme.name())
        fig_cc.savefig(path.join(savedir, 'centroids.png'))
        fig_s.savefig(path.join(savedir, 'silhouettes.png'))
