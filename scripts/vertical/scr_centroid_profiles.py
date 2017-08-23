# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir

case_set = '14-16by_hand'
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)

if __name__ == '__main__':
    plt.ioff()
    #plt.close('all')
    results_dir = ensure_dir(path.join(RESULTS_DIR, 'centroid_profiles', name))
    cases = case.read_cases(case_set)
    c = cases.case.loc['140221-22']
    c.load_classification(name)
    #c.load_pluvio()
    #c.plot(cmap='viridis')
    for cla in c.class_scheme.get_class_list():
        axarr = c.plot_centroid(cla)
        fig = axarr[0].figure
        fig.savefig(path.join(results_dir, 'class{0:02d}'.format(cla)),
                              bbox_inches='tight')
        plt.close(fig)

