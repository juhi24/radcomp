# coding: utf-8
"""classification script for both snow and rain events"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, RESULTS_DIR
from j24 import ensure_join
from warnings import warn
import conf

save = True


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    case_set = 'erad18_snow'
    name = conf.SCHEME_ID_SNOW
    cases = multicase.read_cases(case_set)
    #cases = cases[cases.ml_ok.astype(bool)]
    results_dir = ensure_join(RESULTS_DIR, 'classified', name, case_set)
    #c = cases.case['140303']
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
        fig, axarr = c.plot(n_extra_ax=0, plot_snd=False, plot_t=True,
                            plot_lwe=False, cmap='viridis')
        if save:
            fname = path.join(results_dir, c.name()+'.png')
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
