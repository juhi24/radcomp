# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from datetime import timedelta
from j24 import home, ensure_join


CACHE_DIR = ensure_join(home(), '.pysonde', 'cache')
CACHE_KEY_FMT = 'wyo%Y%m%d%H'


def round_hours(timestamp, hres=12):
    """round timestamp to hres hours"""
    tt = timestamp + timedelta(hours=hres/2)
    dt = timedelta(hours=tt.hour%hres, minutes=tt.minute, seconds=tt.second)
    return tt - dt


def sounding_url(t, dtype='text'):
    out_type = dict(pdf='PDF%3ASTUVE', text='TEXT%3ASIMPLE')
    baseurl = 'http://weather.uwyo.edu/cgi-bin/sounding'
    query = '?region=europe&TYPE={type}&YEAR={year}&MONTH={month:02d}&FROM={day:02d}{hour:02d}&TO={day:02d}{hour:02d}&STNM=02963'
    urlformat = baseurl + query
    return urlformat.format(type=out_type[dtype], year=t.year, month=t.month,
                            day=t.day, hour=t.hour)


def read_sounding(timestamp, index_col=0, caching=True, **kws):
    """read wyoming sounding with optional caching (default)"""
    if caching:
        if in_cache(timestamp, **kws):
            data = cache_read(timestamp, **kws)
            # TODO: index col swapping
            return data
    skiprows = (0,1,2,3,4,5,6,8,9)
    url = sounding_url(timestamp)
    data = pd.read_table(url, delim_whitespace=True, index_col=index_col, skiprows=skiprows).dropna()
    data = data.drop(data.tail(1).index).astype(np.float)
    data.index = data.index.astype(np.float)
    if caching:
        cache_write(timestamp, data, **kws)
    return data


def cache_file(station='02963', cachedir=CACHE_DIR):
    """cache file path"""
    return path.join(cachedir, station+'.h5')


def cache_filename_key(timestamp, **kws):
    """oneliner to get both cache filename and key"""
    filename = cache_file(**kws)
    key = timestamp.strftime(CACHE_KEY_FMT)
    return filename, key


def in_cache(timestamp, **kws):
    """check if sounding"""
    filename, key = cache_filename_key(timestamp, **kws)
    if not path.exists(filename):
        return False
    with pd.HDFStore(filename, mode='r') as store:
        if '/'+key in store.keys():
            return True
    return False


def cache_write(timestamp, data, **kws):
    """sounding to cache"""
    data.to_hdf(*cache_filename_key(timestamp, **kws))


def cache_read(timestamp, **kws):
    """Read sounding from cache."""
    return pd.read_hdf(*cache_filename_key(timestamp, **kws))


def create_pn(freq='12H'):
    dt_start = pd.datetime(2015, 11, 1, 00)
    dt_end = pd.datetime(2016, 4, 1, 00)
    dt_range = pd.date_range(dt_start, dt_end, freq=freq)
    d = {}
    for dt in dt_range:
        print(str(dt))
        try:
            d[dt] = read_sounding(dt)
        except Exception as e:
            print(str(dt) + ': ' + str(e))
    pn = pd.Panel(d).transpose(2,1,0)
    return pn


def create_hdf(filepath='/home/jussitii/DATA/soundings.hdf', key='w1516'):
    with pd.HDFStore(filepath) as store:
        store[key] = create_pn()


def reindex(data, force_asc=False):
    i=np.array(range(int(data.index.min()*10),int(data.index.max()*10+1)))/10.
    ii = i[::-1]
    if force_asc:
        return data.reindex(i)
    return data.reindex(ii)


def interp_akima(data):
    data = reindex(data, force_asc=True)
    intrp = data.interpolate(method='akima')
    return intrp.sort_index(ascending=False)


def comp_interp_methods(rh):
    methods = ('linear', 'index', 'values', 'nearest', 'zero','slinear', 'quadratic', 'cubic', 'barycentric', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima')
    for method in methods:
        interp = rh.interpolate(method=method)
        plt.figure()
        interp.plot(label=method)
        plt.legend()
