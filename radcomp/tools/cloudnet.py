# coding: utf-8
"""methods for working with cloudnet model data"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import xarray as xr


def ds2df(ds, heights=None, times=None, variable='temperature'):
    """"
    model data from Dataset to DataFrame interpolating to given resolution
    """
    xh = ds['height']
    if heights is None:
        heights = xh.to_dataframe()['height'].unstack().mean().astype(int)
    xvar = ds[variable]
    hmap = np.empty((xh.shape[0], heights.size))
    for i, hrow in enumerate(xh):
        hmap[i] = np.interp(heights, hrow, ds.level)
    coords = {'time': ds.time, 'h': heights}
    xmap = xr.DataArray(hmap, dims=['time', 'h'], coords=coords)
    xvar_interp = xvar.interp(level=xmap, time=ds.time)
    if times is not None:
        xvar_interp = xvar_interp.interp(time=times)
    return xvar_interp.to_dataframe()[variable].unstack().T

