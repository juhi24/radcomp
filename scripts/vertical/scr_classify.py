#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir

plt.ion()
plt.close('all')
np.random.seed(0)

#dt0 = pd.datetime(2014, 2, 21, 12, 30)
#dt1 = pd.datetime(2014, 2, 22, 15, 30)
case_set = 'training'
n_eigens = 20
n_clusters = 20
reduced = True
save = True

def prep_case(dt0, dt1, n_comp=20):
    c = case.Case.from_dtrange(dt0, dt1)
    c.load_classification('2014rhi_{n}comp'.format(n=n_comp))
    return c

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='baecc+1415', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
#name = '2014rhi_{n}comp'.format(n=n_eigens)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classified', name, case_set))
#c = cases.case['140303']
for i, c in cases.case.iteritems():
    c.load_classification(name)
    #c.plot_classes()
    fig, axarr = c.plot(cmap='viridis', n_extra_ax=0)
    if save:
        fig.savefig(path.join(results_dir, c.name()+'.png'))
