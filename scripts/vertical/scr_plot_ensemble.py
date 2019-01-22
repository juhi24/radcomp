# coding: utf-8
"""Development of VP ensemble plot"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, plotting, case

import conf


def t_boxplot(c):
    t_cl = pd.concat([c.t_surface(), c.classes], axis=1)
    ax = t_cl.boxplot(by='class')
    ax.set_title('Surface temperature distribution by class')
    ax.set_xlabel('class')
    ax.set_ylabel('$T_s$')
    ax.get_figure().suptitle('')
    return ax


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases_id = 'snow'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    #c = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag, has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    #c.load_classification(name)
    for cl in c.class_scheme.get_class_list():
        dat = case.fillna(c.data.loc[('zh', 'kdp', 'zdr'), :10000, c.classes==cl])
        axarr = plotting.plot_vps(dat.iloc[:,:,0], linewidth=0.5, alpha=0)
        for dt in dat.minor_axis:
            plotting.plot_vps(dat.loc[:,:,dt], linewidth=0.5, alpha=0.05,
                              color='blue', axarr=axarr)
        q1 = dat.apply(lambda df: df.quantile(q=0.25), axis=2)
        q3 = dat.apply(lambda df: df.quantile(q=0.75), axis=2)
        med = dat.median(axis=2)
        cen = c.clus_centroids()[0].loc[:,:,cl]
        plotting.plot_vps(med, color='yellow', axarr=axarr)
        plotting.plot_vps(cen, color='red', axarr=axarr)
        plotting.plot_vps_betweenx(q1, q3, axarr=axarr, alpha=0.5)
        fig = axarr[0].get_figure()
        fig.suptitle('Class {}'.format(cl))
    t_boxplot(c)
