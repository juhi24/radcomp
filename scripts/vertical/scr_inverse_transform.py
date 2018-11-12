# coding: utf-8
"""inverse transform experiments for selecting number of PCA components"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, plotting, case

import conf


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    c.load_classification(conf.SCHEME_ID_MELT)
    pn = c.class_scheme.inverse_transform()
    pn.major_axis = c.cl_data_scaled.minor_axis
    pn.minor_axis = c.data.minor_axis
    decoded = case.scale_data(pn, reverse=True)
    c.plot(cmap='viridis', plot_silh=False, above_ml_only=True)
    plotting.plotpn(decoded, cmap='viridis')