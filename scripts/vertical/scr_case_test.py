# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import matplotlib.pyplot as plt
from radcomp.vertical import case, classification

CASE_SET = '14-16by_hand'
n_eigens = 25
n_clusters = 20
reduced = True

NAME = classification.scheme_name(basename='14-16_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    np.random.seed(0)
    cases = case.read_cases(CASE_SET)
    c = cases.case.iloc[0]
    c.load_classification(NAME)
    c.load_pluvio()
    c.plot(cmap='viridis')


