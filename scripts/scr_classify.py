#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from radcomp.vertical import case, read_cases

plt.ion()
plt.close('all')
np.random.seed(0)

#dt0 = pd.datetime(2014, 2, 21, 12, 30)
#dt1 = pd.datetime(2014, 2, 22, 15, 30)
n_comp = 20

def prep_case(dt0, dt1, n_comp=20):
    c = case.Case.from_dtrange(dt0, dt1)
    c.load_classification('2014rhi_{n}comp'.format(n=n_comp))
    return c

cases = read_cases('analysis')
c = cases.case['mar3']
scheme = '2014rhi_{n}comp'.format(n=n_comp)
c.load_classification(scheme)
#c.plot_classes()
fig, axarr = c.plot(cmap='viridis', n_extra_ax=0)
