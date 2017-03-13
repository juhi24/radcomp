# coding: utf-8
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp import RESULTS_DIR
from radcomp.vertical import case, classification, read_cases

plt.ion()
plt.close('all')
np.random.seed(0)

plot = False

params = ['ZH', 'zdr', 'kdp']
hmax = 10000
n_eigens = 20

cases = read_cases('training')

if plot:
    for name, c in cases.case.iteritems():
        fig, axarr = case.plot(params=params, cmap='viridis')
        savepath = path.join(RESULTS_DIR, 'cases', name+'.png')
        fig.savefig(savepath)

pns = [c.data for i,c in cases.case.iteritems()]
data = pd.concat(pns, axis=2)
scheme = classification.VPC(params=params, hmax=hmax, n_eigens=n_eigens)
c = case.Case(data=data, class_scheme=scheme)
c.scale_cl_data()
c.train()
scheme.save('2014rhi_{}comp'.format(n_eigens))

