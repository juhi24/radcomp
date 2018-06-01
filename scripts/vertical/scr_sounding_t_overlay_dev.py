# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import multicase


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    case_set = 'mlt_test'
    name = 'mlt_18eig17clus_pca'
    cases = multicase.read_cases(case_set)
    cases = cases[cases.ml_ok.astype(bool)]
    c = cases.case.iloc[0]
    c.load_classification(name)
    fig, axarr = c.plot(cmap='viridis')
    #xarr[2].contour(t.index, t.columns, t.T, levels=[0], colors='pink')
    #axarr[2].contour(t.columns, t.index, t, levels=[-8, -3], colors='red')
    #axarr[2].contour(t.columns, t.index, t, levels=[-22], colors='olive')
