# coding: utf-8
"""methods for working with cloudnet model data"""

from os import path
from urllib.request import urlretrieve
from urllib.error import HTTPError

import numpy as np
import xarray as xr

from radcomp.tools import strftime_date_range


BASE_URL = 'http://cloudnet.fmi.fi/cgi-bin/cloudnetdata.cgi'
URL_FMT = BASE_URL + '?date=%Y%m%d&type=model&product={product}&site=hyytiala'
DATA_DIR = path.expanduser('~/DATA/hyde_model')
FILENAME_FMT = '%Y%m%d_{product}_hyde.nc'


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


def download_data(datetimes, datadir=DATA_DIR, product='gdas1',
                  overwrite=False):
    """Download model data for given dates."""
    for t in datetimes:
        url = t.strftime(URL_FMT).format(product=product)
        filename = t.strftime(FILENAME_FMT).format(product=product)
        dest = path.join(datadir, product, filename)
        if path.exists(dest) and not overwrite:
            continue
        print(dest)
        try:
            urlretrieve(url, dest)
        except HTTPError as e:
            print(e)


def ds_date_range(t_start, t_end, datadir=DATA_DIR, product='gdas1', **kws):
    """Load model data from a date range as Dataset."""
    filename_fmt = FILENAME_FMT.format(product=product)
    filepath_fmt = path.join(datadir, product, filename_fmt)
    filepaths = strftime_date_range(t_start, t_end, filepath_fmt)
    ds = xr.auto_combine([xr.open_dataset(fp) for fp in filepaths], **kws)
    _, i = np.unique(ds.time, return_index=True)
    return ds.isel(time=i)


def load_as_df(heights, times, **kws):
    """Load model data as DataFrame."""
    ds = ds_date_range(times[0], times[-1])
    return ds2df(ds, heights=heights, times=times, **kws)
