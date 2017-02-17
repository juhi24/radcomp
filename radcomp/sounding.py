#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sounding_url(datetime):
    year = datetime.year
    mon = str(datetime.month).zfill(2)
    day = str(datetime.day).zfill(2)
    hour = str(datetime.hour).zfill(2)
    urlformat = 'http://weather.uwyo.edu/cgi-bin/sounding?region=europe&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}{hour}&TO={day}{hour}&STNM=02963'
    return urlformat.format(year=year, month=mon, day=day, hour=hour)

def read_sounding(datetime):
    skiprows = (0,1,2,3,4,5,6,8,9)
    url = sounding_url(datetime)
    data = pd.read_table(url, delim_whitespace=True, index_col=0, skiprows=skiprows).dropna()
    data = data.drop(data.tail(1).index).astype(np.float)
    data.index = data.index.astype(np.float)
    return data

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
