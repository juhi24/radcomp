#!/usr/bin/env python2
# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir

plt.ion()
plt.close('all')
np.random.seed(0)

case_set = 'test'
n_eigens = 25
n_clusters = 20
reduced = True
save = True

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classified', name, case_set))
#c = cases.case['140303']
for i, c in cases.case.iteritems():
    c.load_classification(name)
    #c.plot_classes()
    fig, axarr = c.plot(n_extra_ax=0)
    if save:
        fig.savefig(path.join(results_dir, c.name()+'.png'))

