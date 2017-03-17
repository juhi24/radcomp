#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, RESULTS_DIR
from j24 import ensure_dir

plt.ioff()
plt.close('all')
np.random.seed(0)

#dt0 = pd.datetime(2014, 2, 21, 12, 30)
#dt1 = pd.datetime(2014, 2, 22, 15, 30)
n_comp = 20
save = True

def prep_case(dt0, dt1, n_comp=20):
    c = case.Case.from_dtrange(dt0, dt1)
    c.load_classification('2014rhi_{n}comp'.format(n=n_comp))
    return c

cases = case.read_cases('all')
scheme = '2014rhi_{n}comp'.format(n=n_comp)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classified'))
#c = cases.case['140303']
for i, c in cases.case.iteritems():
    c.load_classification(scheme)
    #c.plot_classes()
    fig, axarr = c.plot(cmap='viridis', n_extra_ax=0)
    if save:
        fig.savefig(path.join(results_dir, c.name()+'.png'))
