# coding: utf-8
"""Development of VP ensemble plot"""

from os import path

import pandas as pd
import matplotlib.pyplot as plt

import radcomp.visualization as vis
from radcomp.vertical import multicase, plotting, case, RESULTS_DIR
from j24 import ensure_join

import conf


def lineboxplot(dat, rain_season, cen=None):
    """Plot individual profile lines, quantiles and optionally centroid."""
    axarr = plotting.plot_vps(dat.iloc[:,:,0], linewidth=0.5, alpha=0)
    kws = dict(has_ml=rain_season, axarr=axarr)
    for dt in dat.minor_axis:
        plotting.plot_vps(dat.loc[:,:,dt], linewidth=0.5, alpha=0.05,
                          color='blue', **kws)
    q1 = dat.apply(lambda df: df.quantile(q=0.25), axis=2)
    q3 = dat.apply(lambda df: df.quantile(q=0.75), axis=2)
    med = dat.median(axis=2)
    plotting.plot_vps(med, color='yellow', label='median', **kws)
    if cen is not None:
        cenn = cen.reindex_like(med) # add missing columns
        plotting.plot_vps(cenn, color='darkred', label='centroid', **kws)
    plotting.plot_vps_betweenx(q1, q3, alpha=0.5, label='50%', **kws)
    axarr[-1].legend()
    return axarr


def lineboxplots(c, rain_season, fields=None, xlim_override=False, savedir=None):
    """Plot line boxplots of profiles for each class"""
    fields = fields or ('kdp', 'zdr', 'zh') # alphabet order for xlim_override
    data = c.data_above_ml
    axarrlist = []
    seasonchar = 'r' if rain_season else 's'
    fname_extra = 'i' if xlim_override else ''
    if 'T' in fields:
        fname_extra += 't'
    fnamefmt = '{seasonchar}{cl:02.0f}.png'
    titlefmt = 'Class {seasonchar}{cl}'
    for cl in c.vpc.get_class_list():
        dat = case.fillna(data.loc[fields, :10000, c.vpc.classes==cl])
        cen = c.vpc.clus_centroids()[0].loc[:,:,cl]
        axarr = lineboxplot(dat, rain_season, cen=cen)
        if xlim_override:
            for i, ax in enumerate(axarr):
                plotting.vp_xlim(fields[i], ax, not rain_season)
        if savedir is not None:
            fig = axarr[0].get_figure()
            fig.suptitle(titlefmt.format(seasonchar=seasonchar, cl=cl))
            fname = fnamefmt.format(seasonchar=seasonchar+fname_extra, cl=cl)
            fig.savefig(path.join(savedir, fname))
        axarrlist.append(axarr)
    return axarrlist


if __name__ == '__main__':
    plt.ioff()
    plt.close('all')
    cases_id = 'snow'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    c = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag, has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    c.load_classification(name)
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name, 'class_vp_ensemble')
    axarrlist = lineboxplots(c, rain_season, savedir=savedir, fields=('kdp', 'zdr', 'zh', 'T'))
    if not rain_season:
        fig, ax, lines = plotting.boxplot_t_surf(c)
        fig.savefig(path.join(savedir, 't_boxplot.png'), bbox_inches='tight')
