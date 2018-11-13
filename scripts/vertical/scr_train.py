# coding: utf-8
"""
A script for using the VP classification training method
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification, RESULTS_DIR
from j24 import ensure_join
from conf import VPC_PARAMS_SNOW


save_plots = True
train = True
save_scheme = True

VPC_PARAMS_SNOW.update({'n_clusters': 19,
                        'radar_weights': {'kdp': 1.7},
                        'n_eigens': 30})


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    np.random.seed(1)
    cases = multicase.read_cases('snow')
    scheme = classification.VPC(**VPC_PARAMS_SNOW)
    c = multicase.MultiCase.by_combining(cases, class_scheme=scheme)
    if train:
        c.train()
        if save_scheme:
            scheme.save()
    # Load classification and plot centroids
    c.load_classification()
    fig, axarr, i = c.plot_cluster_centroids(colorful_bars='blue')
    ax_sca = c.scatter_class_pca(plot3d=True)
    fig_s, ax_s = plt.subplots()
    c.plot_silhouette(ax=ax_s, cols=(0, 1, 2))#, 'temp_mean'))
    if save_plots:
        savedir = ensure_join(RESULTS_DIR, 'classes_summary', c.class_scheme.name())
        savefile = path.join(savedir, 'centroids.png')
        fig.savefig(savefile, bbox_inches='tight')
