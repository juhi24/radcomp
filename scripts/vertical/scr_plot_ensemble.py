# coding: utf-8
"""Development of VP ensemble plot"""

from os import path

import pandas as pd
import matplotlib.pyplot as plt

from radcomp.vertical import multicase, plotting, case, RESULTS_DIR
from j24 import ensure_join

import conf


def lineboxplot(dat, cen=None):
    """Plot individual profile lines, quantiles and optionally centroid."""
    axarr = plotting.plot_vps(dat.iloc[:,:,0], linewidth=0.5, alpha=0)
    kws = dict(has_ml=rain_season, axarr=axarr)
    for dt in dat.minor_axis:
        plotting.plot_vps(dat.loc[:,:,dt], linewidth=0.5, alpha=0.05,
                          color='blue', **kws)
    q1 = dat.apply(lambda df: df.quantile(q=0.25), axis=2)
    q3 = dat.apply(lambda df: df.quantile(q=0.75), axis=2)
    med = dat.median(axis=2)
    plotting.plot_vps(med, color='yellow', **kws)
    if cen is not None:
        plotting.plot_vps(cen, color='darkred', **kws)
    plotting.plot_vps_betweenx(q1, q3, alpha=0.5, **kws)
    return axarr


def lineboxplots(c, name, rain_season, savedir=None):
    """Plot line boxplots of profiles for each class"""
    data = c.data_above_ml
    axarrlist = []
    for cl in c.vpc.get_class_list():
        dat = case.fillna(data.loc[('zh', 'kdp', 'zdr'), :10000, c.vpc.classes==cl])
        cen = c.vpc.clus_centroids()[0].loc[:,:,cl]
        axarr = lineboxplot(dat, cen)
        if savedir is not None:
            fig = axarr[0].get_figure()
            fig.suptitle('Class {}'.format(cl))
            fig.savefig(path.join(savedir, '{:02.0f}.png'.format(cl)))
        axarrlist.append(axarr)
    return axarrlist


if __name__ == '__main__':
    plt.ioff()
    plt.close('all')
    cases_id = 'rain'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    c = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag, has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    c.load_classification(name)
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name, 'class_vp_ensemble')
    axarrlist = lineboxplots(c, name, rain_season, savedir)
    if not rain_season:
        fig, ax, lines = plotting.boxplot_t_surf(c)
        fig.savefig(path.join(savedir, 't_boxplot.png'), bbox_inches='tight')
