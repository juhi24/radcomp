# coding: utf-8
"""
A script for using the VP classification training method
"""
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from radcomp.vertical import multicase, classification, benchmark, plotting, RESULTS_DIR
from j24 import ensure_join

from conf import VPC_PARAMS_SNOW, VPC_PARAMS_RAIN


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
    #
    if plt_silh:
        fig_s, ax_s = plt.subplots()
        c.plot_silhouette(ax=ax_s)
    else:
        ax_s = None
    #
    stat = bm_stats(c)
    fig_bm, ax_bm = plt.subplots()
    plotting.plot_bm_stats(stat, ax=ax_bm)
    #
    fig_top, ax_top = plt.subplots()
    plotting.boxplot_t_echotop(c, ax=ax_top)
    #
    if save_plots:
        savekws = {'bbox_inches': 'tight'}
        if savedir is None:
            name = c.class_scheme.name()
            savedir = ensure_join(RESULTS_DIR, 'classes_summary', name)
        fig.savefig(path.join(savedir, 'centroids.png'), **savekws)
        if plt_silh:
            fig_s.savefig(path.join(savedir, 'silhouettes.png'), **savekws)
        fig_bm.savefig(path.join(savedir, 'benchmark.png'), **savekws)
        fig_top.savefig(path.join(savedir, 't_top.png'), **savekws)
    return axarr, ax_sca, ax_s, ax_bm, ax_top


def bm_stats(c):
    bm = benchmark.AutoBenchmark(benchmark.autoref(c.data_above_ml))
    bm.fit(c.class_scheme)
    return bm.query_all(bm.query_count)


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
    cases_id = 'rain'
    #
    rain_season = cases_id in ('rain',)
    cases = multicase.read_cases(cases_id)
    if rain_season:
        cases = cases[cases['ml_ok'].fillna(0).astype(bool)]
    vpc_params = VPC_PARAMS_RAIN if rain_season else VPC_PARAMS_SNOW
    c = multicase.MultiCase.by_combining(cases, has_ml=rain_season)
    if bracketing:
        for n_clusters in np.arange(8, 18):
            vpc_params.update({'n_clusters': n_clusters})
            plot_kws = dict(plt_silh=False, plt_sca=False, save_plots=True)
            workflow(c, vpc_params, plot_kws=plot_kws)
    else:
        kws = dict(save_plots=True)
        workflow(c, vpc_params, plot_kws=kws)
