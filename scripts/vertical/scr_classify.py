# coding: utf-8
"""classification script for both snow and rain events"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import RESULTS_DIR
from j24 import ensure_join
from warnings import warn
import conf

save = False

if __name__ == '__main__':
    plt.ioff() if save else plt.ion()
    plt.close('all')
    rain_season = False
    #case_set = conf.CASES_MELT if rain_season else conf.CASES_SNOW
    case_set = 'mar20'
    name = conf.SCHEME_ID_MELT if rain_season else conf.SCHEME_ID_SNOW
    #name = 'snow_t08_kdp17_18eig19clus_pca'
    plot_t = not rain_season
    cases = conf.init_cases(cases_id=case_set)
    #results_dir = ensure_join(RESULTS_DIR, 'classified', name, case_set)
    results_dir = ensure_join(RESULTS_DIR, 'proc', name, case_set)
    for i, c in cases.case.iteritems():
        print(i)
        try:
            c.load_classification(name)
            try:
                c.load_pluvio()
            except FileNotFoundError:
                warn('Pluvio data not found.')
        except ValueError as e:
            warn(str(e))
            raise e
            continue
        #c.plot_classes()
        #c.plot_cluster_centroids()
        fig, axarr = c.plot(n_extra_ax=0, plot_extras=['ts', 'silh', 'cl'],
                            t_contour_ax_ind='all', t_levels=[-20, -8, -3],
                            fig_scale_factor=0.8, cmap='viridis')
        if save:
            fname = path.join(results_dir, c.name()+'.png')
            #fig.set_size_inches([3.58666667, 5.92666667])
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
            del(c)
