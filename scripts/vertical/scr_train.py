# coding: utf-8
"""
A script for using the VP classification training method
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification, RESULTS_DIR
from j24 import ensure_join
from conf import VPC_PARAMS_SNOW, VPC_PARAMS_RAIN


save_plots = False

VPC_PARAMS_SNOW.update({'n_clusters': 13})
VPC_PARAMS_RAIN.update({'n_clusters': 10})


def training(c, train=True, save_scheme=True):
    if train:
        c.train(quiet=True)
        if save_scheme:
            scheme.save()
    # Load classification and plot centroids
    c.load_classification()


if __name__ == '__main__':
    plt.ion()
    #plt.close('all')
    np.random.seed(1)
    cases_id = 'snow'
    #
    rain_season = cases_id in ('rain',)
    cases = multicase.read_cases(cases_id)
    if rain_season:
        cases = cases[cases.ml_ok.fillna(0).astype(bool)]
    vpc_params = VPC_PARAMS_RAIN if rain_season else VPC_PARAMS_SNOW
    c = multicase.MultiCase.by_combining(cases, has_ml=rain_season)
    for n_clusters in (13,):
        vpc_params.update({'n_clusters': n_clusters})
        scheme = classification.VPC(**vpc_params)
        c.class_scheme = scheme
        training(c, train=True, save_scheme=True)
        fig, axarr, i = c.plot_cluster_centroids(colorful_bars='blue', cmap='viridis')
        ax_sca = c.scatter_class_pca(plot3d=True)
        #fig_s, ax_s = plt.subplots()
        #c.plot_silhouette(ax=ax_s)
        if save_plots:
            savedir = ensure_join(RESULTS_DIR, 'classes_summary', c.class_scheme.name())
            fig.savefig(path.join(savedir, 'centroids.png'), bbox_inches='tight')
            fig_s.savefig(path.join(savedir, 'silhouettes.png'), bbox_inches='tight')
