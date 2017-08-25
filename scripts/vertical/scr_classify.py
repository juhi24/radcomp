# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
"""
@author: Jussi Tiira
"""
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir
from warnings import warn

plt.ioff()
plt.close('all')

case_set = '14-16by_hand'
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

save = True

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classified', name, case_set))
#c = cases.case['140303']
for i, c in cases.case.iteritems():
    print(i)
    try:
        c.load_classification(name)
        c.load_pluvio()
    except ValueError as e:
        warn(str(e))
        continue
    #c.plot_classes()
    fig, axarr = c.plot(n_extra_ax=0, plot_fr=True)#, cmap='viridis')
    if save:
        fig.savefig(path.join(results_dir, c.name()+'.png'), bbox_inches='tight')
        plt.close(fig)

