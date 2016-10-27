#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import netCDF4 as nc
import os
import pandas as pd
import matplotlib.pyplot as plt
import pyart

def ensure_path(directory):
    """Make sure the path exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def ncdatetime(ncdata):
    times = ncdata.variables['time']
    return nc.num2date(times[:], times.units)
    
gridpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/grid/'

datafilepath = os.path.join(gridpath, 'KUM', '20160903', 'ncf_20160903_070054.nc')
datadir = os.path.dirname(datafilepath)
ncdata = nc.Dataset(datafilepath, 'r')
discardir = os.path.join(datadir, 'discarded')
ensure_path(discardir)

if 'KDP' not in list(ncdata.variables):
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(datafilepath))
    os.rename(datafilepath, discardfilepath)
    #continue

dbz = ncdata.variables['DBZ']