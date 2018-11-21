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


VPC_PARAMS_SNOW.update({'n_clusters': 12})
VPC_PARAMS_RAIN.update({'n_clusters': 10, 'invalid_classes': [9]})


def training(c, train=True, save_scheme=True):
    if train:
        c.train(quiet=True)
        if save_scheme:
            c.class_scheme.save()
    c.load_classification()


def make_plots(c, save_plots=False, savedir=None, plt_silh=True, plt_sca=True):
    fig, axarr, i = c.plot_cluster_centroids(colorful_bars='blue',
                                             fig_scale_factor=0.8)#, cmap='viridis')
    ax_sca = c.scatter_class_pca(plot3d=True) if plt_sca else None
    if plt_silh:
        fig_s, ax_s = plt.subplots()
        c.plot_silhouette(ax=ax_s)
    else:
        ax_s = None
    if save_plots:
        if savedir is None:
            name = c.class_scheme.name()
            savedir = ensure_join(RESULTS_DIR, 'classes_summary', name)
        fig.savefig(path.join(savedir, 'centroids.png'), bbox_inches='tight')
        if plt_silh:
            fig_s.savefig(path.join(savedir, 'silhouettes.png'), bbox_inches='tight')
    return axarr, ax_sca, ax_s


def workflow(c, vpc_params, plot_kws={}, **kws):
    """training workflow"""
    scheme = classification.VPC(**vpc_params)
    c.class_scheme = scheme
    training(c, **kws)
    make_plots(c, **plot_kws)


if __name__ == '__main__':
    bracketing = False
    plt.ion()
    plt.close('all')
    np.random.seed(1)
    cases_id = 'snow'
    #
    rain_season = cases_id in ('rain',)
    cases = multicase.read_cases(cases_id)
    if rain_season:
        cases = cases[cases.ml_ok.fillna(0).astype(bool)]
    vpc_params = VPC_PARAMS_RAIN if rain_season else VPC_PARAMS_SNOW
    c = multicase.MultiCase.by_combining(cases, has_ml=rain_season)
    if bracketing:
        for n_clusters in (10, 12, 14, 16, 18, 20):
            vpc_params.update({'n_clusters': n_clusters})
            plot_kws = dict(plt_silh=False, plt_sca=False)
            workflow(c, vpc_params, plot_kws=plot_kws)
    else:
        kws = dict(save_plots=True)
        workflow(c, vpc_params, plot_kws=kws)
