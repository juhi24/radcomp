# coding: utf-8
"""classification script for both snow and rain events"""

import gc
from os import path
from warnings import warn

import matplotlib.pyplot as plt

from radcomp.vertical import RESULTS_DIR
from j24 import ensure_join

import conf


DEBUG = True


if __name__ == '__main__':
    # TODO: weird memory leak with rain cases
    save = True
    plt.ioff() if save else plt.ion()
    plt.close('all')
    rain_season = True
    if DEBUG:
        rain_season = True
        case_set = 'debug_short'
    else:
        case_set = conf.CASES_RAIN if rain_season else conf.CASES_SNOW
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    cases = conf.init_cases(cases_id=case_set)
    if DEBUG:
        results_dir = ensure_join(RESULTS_DIR, 'debug', name, case_set)
    else:
        results_dir = ensure_join(RESULTS_DIR, 'classified', name, case_set)
    for i, c in cases.case.iteritems():
        print(i)
        c.load_classification(name)
        try:
            c.load_pluvio()
        except FileNotFoundError:
            warn('Pluvio data not found.')
        #c.plot_classes()
        #c.plot_cluster_centroids()
        fig, axarr = c.plot(params=['kdp', 'zh', 'zdr'],
                            n_extra_ax=0, plot_extras=['ts', 'silh', 'cl'],
                            t_contour_ax_ind='all',
                            t_levels=[-40, -20, -10, -8, -3],
                            fig_scale_factor=0.75)#, cmap='viridis')
        if save:
            fname = path.join(results_dir, c.name()+'.png')
            #fig.set_size_inches([3.58666667, 5.92666667])
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
            del(c)
            gc.collect()
