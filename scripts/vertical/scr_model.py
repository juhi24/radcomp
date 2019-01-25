# coding: utf-8

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
    c.load_model_data(variable='rh')
    params = ['zh', 'zdr', 'kdp', 'rh']
    fig, axarr = c.plot(plot_fr=False, plot_t=False, plot_azs=False,
                        t_contour_ax_ind=[0, 1, 2], cmap='viridis', params=params,
                        t_levels=(-20, -8, 0))
