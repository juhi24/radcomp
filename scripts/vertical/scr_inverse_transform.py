# coding: utf-8
"""inverse transform experiments for selecting number of PCA components"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, case, plotting

import conf


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    c.load_classification(conf.SCHEME_ID_MELT)
    c.plot(cmap='viridis', plot_silh=False, above_ml_only=True)
    c.plot(cmap='viridis', plot_silh=False, inverse_transformed=True)
    # difference
    ref = c.cl_data.transpose(0,2,1)
    tr = c.inverse_transform()
    diff = (case.fillna(ref).subtract(tr))
    fig, axarr = plotting.plotpn(diff.abs())
    axarr[0].set_title('Absolute error of inverse transformed data')
    rmse = diff.pow(2).mean().mean().pow(0.5)
