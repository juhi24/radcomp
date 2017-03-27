# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, insitu, RESULTS_DIR
from j24 import ensure_dir


plt.ion()
plt.close('all')
np.random.seed(0)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'class_scatter'))
n_comp = 20

cases = case.read_cases('analysis')
g = pd.read_pickle(insitu.TABLE_PKL)
name = '140303'
data_g = g.loc[name]
c = cases.case[name]
scheme = '2014rhi_{n}comp'.format(n=n_comp)
c.load_classification(scheme)
classes = c.classes
classes.index = classes.index.round('1min')
classes.name = 'class'
i = insitu.time_weighted_mean(data_g, rule='15min', param='intensity', base=14)
i.fillna(0, inplace=True)
i.hist(by=classes, sharex=True, sharey=True, bins=[0,0.5,1,1.5,2,2.5])
