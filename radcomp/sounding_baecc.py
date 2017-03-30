# coding: utf-8
import numpy as np
import netCDF4 as nc
import pandas as pd


soundings_f = '/home/jussitii/DATA/arm/sounding/tmpsondewnpnM1.b1.20140121.125200..20140330.172000.custom.cdf'
#s = nc.Dataset(soundings_f)

def time(ncdata, as_np=False):
    """time from ARM sounding netCDF"""
    t0 = ncdata.variables['base_time'][0]
    if as_np:
        return np.array(t0 + ncdata.variables['time_offset'][:]).astype('datetime64[s]')
    return pd.to_datetime(t0 + ncdata.variables['time_offset'][:], unit='s')

def nc2df(ncdata, variables=None):
    if variables is None:
        variables = list(ncdata.variables)
        for rmvar in ['base_time', 'time_offset', 'time']:
            variables.remove(rmvar)
    df = pd.DataFrame(index=time(ncdata))
    for var in variables:
        df[var] = ncdata.variables[var]
    return df

def deltat(ncdata):
    tim = time(ncdata)
    t = pd.DataFrame(tim, index=tim)
    dt = t.diff()
    dt.name = 'dt'
    return dt
