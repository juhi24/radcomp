# coding: utf-8
"""Plot demo cases with classification."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase#, RESULTS_DIR
#from j24 import ensure_join
import conf


case_set_r = 'erad18_rain'
case_set_s = 'erad18_snow'
scheme_id_r = conf.SCHEME_ID_MELT
scheme_id_s = conf.SCHEME_ID_SNOW
savedir = conf.P1_FIG_DIR

if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    cases_r = multicase.read_cases(case_set_r)
    cases_s = multicase.read_cases(case_set_s)
    for i, c in cases_r.case.iteritems():
        c.load_classification(scheme_id_r)
        fig, axarr = c.plot(n_extra_ax=0, plot_silh=False)#, cmap='viridis')
        fig.set_size_inches(2.5, 4)
        axarr[-1].set_xticks(axarr[-1].get_xticks()[1::2])
        fig.savefig(path.join(savedir, 'hm_case.png'), bbox_inches='tight')
    for i, c in cases_s.case.iteritems():
        c.load_classification(scheme_id_s)
        fig, axarr = c.plot(n_extra_ax=0, plot_silh=False, plot_fr=False,
                            plot_azs=False)
        fig.set_size_inches(3.5, 4)
        #axarr[-1].set_xticks(axarr[-1].get_xticks()[0::2])
        fig.savefig(path.join(savedir, 'dend2_case.png'), bbox_inches='tight')
