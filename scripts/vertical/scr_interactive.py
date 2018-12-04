# coding: utf-8
"""classification script for both snow and rain events"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from warnings import warn

from radcomp.vertical.cases_plotter import CasesPlotter

import conf

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    rain_season = False
    #case_set = conf.CASES_MELT if rain_season else conf.CASES_SNOW
    case_set = 'tmp'
    name = conf.SCHEME_ID_MELT if rain_season else conf.SCHEME_ID_SNOW
    #name = 'snow_t08_kdp17_18eig19clus_pca'
    plot_t = not rain_season
    cases = conf.init_cases(cases_id=case_set)
    for i, c in cases.case.iteritems():
        print(i)
        try:
            c.load_classification(name)
        except ValueError as e:
            warn(str(e))
            raise e
            continue
    cp = CasesPlotter(cases)
    cp.plot(n_extra_ax=0, plot_t=plot_t, plot_silh=False,
            t_contour_ax_ind='all', t_levels=[-20, -8, -3],
            plot_lwe=False, fig_scale_factor=0.8, cmap='viridis', interactive=False)
