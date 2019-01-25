# coding: utf-8
"""interactive plotting of cases"""

import matplotlib.pyplot as plt
from warnings import warn

from radcomp.vertical.cases_plotter import ProfileMarker

import conf

if __name__ == '__main__':
    plt.ion()
    #plt.close('all')
    rain_season = True
    case_set = conf.CASES_MELT if rain_season else conf.CASES_SNOW
    #case_set = 'tmp'
    name = conf.SCHEME_ID_MELT if rain_season else conf.SCHEME_ID_SNOW
    #name = 'snow_t08_kdp17_18eig19clus_pca'
    cases = conf.init_cases(cases_id=case_set)
    for i, c in cases.case.iteritems():
        print(i)
        try:
            c.load_classification(name)
        except ValueError as e:
            warn(str(e))
            raise e
            continue
    cp = ProfileMarker(cases)
    cp.plot(params=['kdp', 'zh', 'zdr', 'zdrg', 'kdpg'], n_extra_ax=0, plot_extras=['cl'],
            t_contour_ax_ind='all', t_levels=[-20, -10, -8, -3],
            fig_scale_factor=0.8, cmap='viridis', interactive=False)
