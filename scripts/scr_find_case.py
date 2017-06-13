# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from datetime import datetime, timedelta
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import home

plt.ion()
plt.close('all')
np.random.seed(0)

case_set = 'test'
n_eigens = 25
n_clusters = 20
reduced = True

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)

datadir = path.join(home(), 'DATA', 'vprhi')
datefile = path.join(datadir, 'date.list')
dates = pd.read_csv(datefile, dtype=str).iloc[:, 0]
date_start = dates.apply(datetime.strptime, args=['%Y%m%d'])
date_end = date_start + timedelta(days=1)
date_start.name = 'start'
date_end.name = 'end'
date_end.index = date_start
dtpath = path.join(datadir, 'date.csv')
date_end.to_csv(dtpath)
pd.read_hdf(path.join(home(), 'DATA', 't_fmi_14-17.h5'), 'data')
