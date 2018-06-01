# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from radcomp import sounding
from radcomp.vertical import multicase


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    case_set = 'mlt_test'
    name = 'mlt_18eig17clus_pca'
    cases = multicase.read_cases(case_set)
    cases = cases[cases.ml_ok.astype(bool)]
    c = cases.case.iloc[1]
    c.load_classification(name)
    ts = c.timestamps().apply(sounding.round_hours, hres=12).drop_duplicates()
    ts.index = ts.values
    if ts.iloc[0] > c.t_start():
        t0 = ts.iloc[0]-timedelta(hours=12)
        ts[t0] = t0
        ts.sort_index(inplace=True)
    if ts.iloc[-1] < c.t_end():
        t1 = ts.iloc[-1]+timedelta(hours=12)
        ts[t1] = t1
        ts.sort_index(inplace=True)
    a = ts.apply(lambda x: sounding.read_sounding(x, index_col='HGHT')['TEMP'])
    a.interpolate(axis=1, inplace=True)
    na = c.timestamps().apply(lambda x: np.nan)
    na = pd.DataFrame(na).reindex(a.columns, axis=1)
    t = pd.concat([na,a]).sort_index().interpolate(method='time')
    t = t.loc[:, 0:10000].drop(a.index)
    fig, axarr = c.plot(cmap='viridis')
    #xarr[2].contour(t.index, t.columns, t.T, levels=[0], colors='pink')
    axarr[2].contour(t.index, t.columns, t.T, levels=[-8, -3], colors='red')
    axarr[2].contour(t.index, t.columns, t.T, levels=[-22], colors='olive')
