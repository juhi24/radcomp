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
save_scheme = True

cases = multicase.read_cases('snow')
vpc_params = dict(basename='snow',
                  params=['ZH', 'zdr', 'kdp'],
                  hlimits=(190, 10e3),
                  n_eigens=22,
                  n_clusters=23,
                  reduced=True,
                  use_temperature=True,
                  t_weight_factor=0.8,
                  radar_weight_factors=dict(kdp=1.3))


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    np.random.seed(1)
    scheme = classification.VPC(**vpc_params)
    c = multicase.MultiCase.by_combining(cases, class_scheme=scheme)
    if train:
        c.train()
        if save_scheme:
            scheme.save()
    # Load classification and plot centroids
    c.load_classification()
    fig, axarr, i = c.plot_cluster_centroids(colorful_bars='blue')
    c.scatter_class_pca(plot3d=True)
    fig_s, ax_s = plt.subplots()
    c.plot_silhouette(ax=ax_s, cols=(0, 1, 2))#, 'temp_mean'))
    if save_plots:
        savedir = ensure_join(RESULTS_DIR, 'classes_summary', c.class_scheme.name())
        savefile = path.join(savedir, 'centroids.png')
        fig.savefig(savefile, bbox_inches='tight')
