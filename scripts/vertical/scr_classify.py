# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
"""
@author: Jussi Tiira
"""
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification, RESULTS_DIR
from j24 import ensure_join
from warnings import warn
import conf

plt.ion()
plt.close('all')

case_set = 'erad18_snow'
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

save = True


if __name__ == '__main__':
    cases = multicase.read_cases(case_set)
    #cases = cases[cases.ml_ok.astype(bool)]
    #name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
    #                                  n_clusters=n_clusters, reduced=reduced,
    #                                  use_temperature=use_temperature,
    #                                  t_weight_factor=t_weight_factor,
    #                                  radar_weight_factors=radar_weight_factors)
    name = conf.SCHEME_ID_SNOW
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
            continue
        #c.plot_classes()
        fig, axarr = c.plot(n_extra_ax=0, plot_snd=False, plot_t=True,
                            plot_lwe=False, cmap='viridis')
        if save:
            fname = path.join(results_dir, c.name()+'.png')
            fig.savefig(fname, bbox_inches='tight')
            #plt.close(fig)

