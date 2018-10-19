# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from radcomp.vertical import multicase

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = multicase.read_cases('t_model')
    c = cases.case[0]
    fname = path.expanduser('~/DATA/hyde_model/gdas1/20140612_hyytiala_gdas1.nc')
    ds = xr.open_dataset(fname)
    #t = ds['temperature'].to_series().unstack().T
    hh = c.data.major_axis
    h = ds['height']
    t = ds['temperature']
    hmap = np.empty((9, hh.size))
    for i, hrow in enumerate(h):
        hmap[i] = np.interp(hh, hrow, ds.level)
    xmap = xr.DataArray(hmap, dims=['time', 'h'], coords={'time':ds.time, 'h':hh})
    temperature = t.interp(level=xmap, time=ds.time).interp(time=c.data.minor_axis)
    plt.figure()
    temperature.T.plot()
    c.data['T'] = temperature.to_dataframe()['temperature'].unstack().T
    c.plot(plot_fr=False, plot_t=False, plot_azs=False,
           plot_snd=False, cmap='viridis', params=['zh', 'zdr', 'kdp', 'T'])