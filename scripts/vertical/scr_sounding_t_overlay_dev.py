# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radcomp import sounding
from radcomp.vertical import multicase


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    case_set = 'mlt_test'
    name = 'mlt_18eig17clus_pca'
    cases = multicase.read_cases(case_set)
    cases = cases[cases.ml_ok.astype(bool)]
    c = cases.case.iloc[0]
    ts = c.timestamps().apply(sounding.round_hours, hres=12).drop_duplicates()
    ts.index = ts.values
    a = ts.apply(lambda x: sounding.read_sounding(x, index_col='HGHT')['TEMP'])
    a.interpolate(axis=1, inplace=True)
    na = c.timestamps().apply(lambda x: np.nan)
    na = pd.DataFrame(na).reindex(a.columns, axis=1)
    t = pd.concat([na,a]).sort_index().interpolate(method='time')
    t = t.loc[:, 0:10000].drop(a.index)
    fig, axarr = c.plot(cmap='viridis')
    axarr[2].contour(t.index, t.columns, t.T, levels=[-8, -3], colors='red')
    axarr[2].contour(t.index, t.columns, t.T, levels=[-22, -10], colors='olive')
