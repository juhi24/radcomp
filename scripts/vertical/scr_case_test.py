# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import case, classification

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
    plt.ion()
    #plt.close('all')
    cases = case.read_cases(case_set)
    c = cases.case.loc['141217']
    c.load_classification(name)
    c.load_pluvio()
    c.plot(cmap='viridis')


