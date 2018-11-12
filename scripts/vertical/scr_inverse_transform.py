# coding: utf-8
"""inverse transform experiments for selecting number of PCA components"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase

import conf


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    c.load_classification(conf.SCHEME_ID_MELT)
    c.plot(cmap='viridis', plot_silh=False, above_ml_only=True)
    c.plot(cmap='viridis', plot_silh=False, inverse_transformed=True)