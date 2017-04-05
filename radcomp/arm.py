# coding: utf-8
import numpy as np
import netCDF4 as nc
import pandas as pd
from glob import glob
from datetime import datetime
from os import path
from j24 import home


SOUNDING_DIR = path.join(home(), 'DATA', 'arm', 'sounding')
GROUND_DIR = gdir=path.join(home(), 'DATA', 'arm', 'ground')
all_soundings_f = 'tmpsondewnpnM1.b1.20140121.125200..20140330.172000.custom.cdf'
sounding_f = 'tmpsondewnpnM1.b1.20140131.115000.cdf' # sample
all_soundings_path = path.join(SOUNDING_DIR, all_soundings_f)
sounding_path = path.join(SOUNDING_DIR, sounding_f)
SOUNDING_GLOB = path.join(SOUNDING_DIR, 'tmpsondewnpnM1.b1.20??????.??????.cdf')
GROUND_GLOB = path.join(GROUND_DIR, 'tmpmetM1.b1.20??????.??????.cdf')
#s = nc.Dataset(soundings_f)

def time(ncdata, as_np=False):
    """time from ARM netCDF"""
    t0 = ncdata.variables['base_time'][0]
    if as_np:
        return np.array(t0 + ncdata.variables['time_offset'][:]).astype('datetime64[s]')
    return pd.to_datetime(t0 + ncdata.variables['time_offset'][:], unit='s')

def nc2df(ncdata, index='time', variables=None):
    """ARM netCDF to dataframe"""
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
    df.drop(index, axis=1, inplace=True)
    return df

def deltat(ncdata):
    tim = time(ncdata)
    t = pd.DataFrame(tim, index=tim)
    dt = t.diff()
    dt.name = 'dt'
    return dt

def path2t(sounding_path):
    fname = path.basename(sounding_path)
    dtstr = ''.join(fname.split('.')[-3:-1])
    return datetime.strptime(dtstr, '%Y%m%d%H%M%S')

def datalist(globfmt=GROUND_GLOB):
    sounding_files = pd.Series(glob(globfmt))
    t = sounding_files.apply(path2t)
    ncs = sounding_files.apply(nc.Dataset)
    ncs.index = t
    ncs.name = 'dataset'
    return ncs.sort_index()

def nearest(i, df=None, **kws):
    if df is None:
        df = datalist(**kws)
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
    
def resampled_t_dp(ncdata):
    df = nc2df(ncdata).loc[:,['tdry','dp']]
    return resample_numeric(df).loc[200:7000]

def df2series(df):
    return df.T.stack()

def prep4pca(df):
    return df.apply(lambda x: df2series(resampled_t_dp(x))).dropna()

def var_in_timerange(tstart, tend, var='temp_mean'):
    gnc = datalist(globfmt=GROUND_GLOB)
    gnc.index = gnc.index.map(lambda t: t.date())
    ncs = gnc.loc[tstart.date():tend.date()]
    t = pd.concat(map(lambda x: nc2df(x, index='time')[var], ncs))
    return t.loc[tstart:tend]


