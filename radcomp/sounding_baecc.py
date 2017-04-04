# coding: utf-8
import numpy as np
import netCDF4 as nc
import pandas as pd
from glob import glob
from datetime import datetime
from os import path
from j24 import home


SOUNDING_DIR = path.join(home(), 'DATA', 'arm', 'sounding')
all_soundings_f = 'tmpsondewnpnM1.b1.20140121.125200..20140330.172000.custom.cdf'
sounding_f = 'tmpsondewnpnM1.b1.20140131.115000.cdf' # sample
all_soundings_path = path.join(SOUNDING_DIR, all_soundings_f)
sounding_path = path.join(SOUNDING_DIR, sounding_f)
#s = nc.Dataset(soundings_f)

def time(ncdata, as_np=False):
    """time from ARM sounding netCDF"""
    t0 = ncdata.variables['base_time'][0]
    if as_np:
        return np.array(t0 + ncdata.variables['time_offset'][:]).astype('datetime64[s]')
    return pd.to_datetime(t0 + ncdata.variables['time_offset'][:], unit='s')

def nc2df(ncdata, index='alt', variables=None):
    if variables is None:
        variables = list(ncdata.variables)
        for rmvar in ['base_time', 'time_offset', 'time']:
            variables.remove(rmvar)
    df = pd.DataFrame()
    t = pd.Series(time(ncdata))
    df['time'] = t
    for var in variables:
        df[var] = ncdata.variables[var]
    df.index = df[index]
    return df

def deltat(ncdata):
    tim = time(ncdata)
    t = pd.DataFrame(tim, index=tim)
    dt = t.diff()
    dt.name = 'dt'
    return dt

def path2t(sounding_path):
    fname = path.basename(sounding_path)
    return datetime.strptime(fname, 'tmpsondewnpnM1.b1.%Y%m%d.%H%M%S.cdf')

def datalist():
    sounding_files = pd.Series(glob(path.join(SOUNDING_DIR, 'tmpsondewnpnM1.b1.20??????.??????.cdf')))
    t = sounding_files.apply(path2t)
    ncs = sounding_files.apply(nc.Dataset)
    ncs.index = t
    ncs.name = 'dataset'
    return ncs.sort_index()

def nearest(i):
    df = datalist()
    return df.iloc[np.argmin(np.abs(df.index - pd.Timestamp(i)))]

def mdf():
    """multi-indexed dataframe of all sounding data"""
    dfs = datalist().apply(nc2df)
    return pd.concat(dfs.values, keys=dfs.index)

def anyround(x, prec=100):
    return round(x/prec)*prec

def resample_numeric(df, **kws):
    """Resample dataframe with numeric index"""
    alt = pd.Series(df.index, index=df.index)
    ralt=alt.apply(anyround, **kws)
    return df.groupby(by=ralt).mean()
    
