# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path

import matplotlib.pyplot as plt

from radcomp.tools import cloudnet
from radcomp.vertical import multicase

import conf


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    c.load_model_temperature()
    #c.load_classification(conf.SCHEME_ID_MELT)
    params = ['zh', 'zdr', 'kdp']
    fig, axarr = c.plot(plot_fr=False, plot_t=False, plot_azs=False,
                        plot_t_contour=True, cmap='viridis', params=params,
                        t_levels=(-20, -8))
