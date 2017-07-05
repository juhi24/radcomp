# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import matplotlib.pyplot as plt
from radcomp.vertical import case, classification

case_set = '14-16by_hand'
n_eigens = 25
n_clusters = 20
reduced = True
use_temperature = True
t_weight_factor = 0.3

name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor)

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    np.random.seed(0)
    cases = case.read_cases(case_set)
    c = cases.case.iloc[0]
    c.load_classification(name)
    c.load_pluvio()
    c.plot(cmap='viridis')


