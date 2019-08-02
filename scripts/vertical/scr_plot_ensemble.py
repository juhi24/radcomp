# coding: utf-8
"""Development of VP ensemble plot"""

from os import path

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, plotting, case, RESULTS_DIR
from j24 import ensure_join

import conf
from plot_wrappers import SAVE_KWS


def lineboxplots(c, fields=None, xlim_override=False, savedir=None):
    """Plot line boxplots of profiles for each class"""
    fields = fields or ('kdp', 'zdr', 'zh') # alphabet order for xlim_override
    rain_season = c.has_ml
    data = c.data_above_ml
    axarrlist = []
    seasonchar = 'R' if rain_season else 'S'
    fname_extra = 'i' if xlim_override else ''
    if 'T' in fields:
        fname_extra += 't'
    fnamefmt = '{seasonchar}{cl:02.0f}.png'
    titlefmt = 'Class {seasonchar}{cl}'
    for cl in c.vpc.get_class_list():
        dat = case.fillna(data.loc[fields, :10000, c.vpc.classes==cl])
        cen = c.vpc.clus_centroids()[0].loc[:,:,cl]
        axarr = plotting.lineboxplot(dat, rain_season, cen=cen)
        if xlim_override:
            for i, ax in enumerate(axarr):
                plotting.vp_xlim(fields[i], ax, not rain_season)
        if savedir is not None:
            fig = axarr[0].get_figure()
            fig.suptitle(titlefmt.format(seasonchar=seasonchar, cl=cl))
            fname = fnamefmt.format(seasonchar=seasonchar+fname_extra, cl=cl)
            fpath = path.join(savedir, fname)
            print(fpath)
            fig.savefig(fpath, **SAVE_KWS)
        axarrlist.append(axarr)
    return axarrlist


if __name__ == '__main__':
    plt.ioff()
    plt.close('all')
    cases_id = 'snow'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    c = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag,
                                          has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    c.load_classification(name)
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name,
                          'class_vp_ensemble')
    axarrlist = lineboxplots(c, savedir=savedir,
                             #xlim_override=True,
                             fields=('kdp', 'zdr', 'zh', 'T'))
    if not rain_season:
        fig, ax, lines = plotting.boxplot_t_surf(c)
        fig.savefig(path.join(savedir, 't_boxplot.png'), **SAVE_KWS)
