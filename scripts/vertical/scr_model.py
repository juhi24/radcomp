# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path

import matplotlib.pyplot as plt

from radcomp.vertical import multicase


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    fname = path.expanduser('~/DATA/hyde_model/gdas1/20140612_hyytiala_gdas1.nc')
    df_t = load_model_data(fname, c.data.major_axis, c.data.minor_axis)
    c.data['T'] = df_t-273.15
    params = ['zh', 'zdr', 'kdp', 'T']
    fig, axarr = c.plot(plot_fr=False, plot_t=False, plot_azs=False,
                        plot_snd=False, cmap='viridis', params=params)
    x = c.data['T']
    for ax in axarr:
        ax.contour(x.columns, x.index, x, levels=[0], colors='dimgray')
