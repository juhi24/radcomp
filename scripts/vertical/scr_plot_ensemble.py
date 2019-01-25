# coding: utf-8
"""Development of VP ensemble plot"""

from os import path

import pandas as pd
import matplotlib.pyplot as plt

from radcomp.vertical import multicase, plotting, case, RESULTS_DIR
from j24 import ensure_join

import conf


def t_boxplot(c, **kws):
    t_cl = pd.concat([c.t_surface(), c.classes], axis=1)
    meanprops = dict(linestyle='-', color='darkred')
    medianprops = dict()
    boxprops = dict()
    fig, ax = plt.subplots(dpi=110, figsize=(5,3.5))
    tup = t_cl.boxplot(by='class', whis=4, return_type='both', showmeans=True,
                       meanline=True, meanprops=meanprops, patch_artist=True,
                       medianprops=medianprops, boxprops=boxprops, ax=ax, **kws)
    ax, lines = tup.temp_mean
    for med in lines['medians']:
        med.set_color('yellow')
    for box in lines['boxes']:
        box.set_alpha(0.5)
    #t_cl.boxplot(by='class', showcaps=False, showbox=False, showfliers=False, ax=ax)
    ax.set_title('Surface temperature distribution by class')
    ax.set_xlabel('class')
    ax.set_ylabel('$T_s$')
    fig = ax.get_figure()
    fig.suptitle('')
    return fig, ax, lines


def lineboxplots(c, name, rain_season, savedir):
    data = c.data_above_ml
    for cl in c.vpc.get_class_list():
        dat = case.fillna(data.loc[('zh', 'kdp', 'zdr'), :10000, c.classes==cl])
        axarr = plotting.plot_vps(dat.iloc[:,:,0], linewidth=0.5, alpha=0)
        kws = dict(has_ml=rain_season, axarr=axarr)
        for dt in dat.minor_axis:
            plotting.plot_vps(dat.loc[:,:,dt], linewidth=0.5, alpha=0.05,
                              color='blue', **kws)
        q1 = dat.apply(lambda df: df.quantile(q=0.25), axis=2)
        q3 = dat.apply(lambda df: df.quantile(q=0.75), axis=2)
        med = dat.median(axis=2)
        cen = c.clus_centroids()[0].loc[:,:,cl]
        plotting.plot_vps(med, color='yellow', **kws)
        plotting.plot_vps(cen, color='darkred', **kws)
        plotting.plot_vps_betweenx(q1, q3, alpha=0.5, **kws)
        fig = axarr[0].get_figure()
        fig.suptitle('Class {}'.format(cl))
        fig.savefig(path.join(savedir, '{:02.0f}.png'.format(cl)))


if __name__ == '__main__':
    plt.ioff()
    plt.close('all')
    cases_id = 'rain'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    c = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag, has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name, 'class_vp_ensemble')
    c.load_classification(name)
    lineboxplots(c, name, rain_season, savedir)
    if not rain_season:
        fig, ax, lines = t_boxplot(c)
        fig.savefig(path.join(savedir, 't_boxplot.png'), bbox_inches='tight')
