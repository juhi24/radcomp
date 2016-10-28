#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import netCDF4 as nc
import os
#import pandas as pd
#import h5py
from pyoptflow import utils, visualization
from pyoptflow.core import extract_motion_proesmans
from pyoptflow.interpolation import interpolate
import matplotlib.pyplot as plt
import numpy as np
#import pyart
#import grid_io_withradx2gridread as gio

def interp(I1, I2, n=10):
    """Interpolate n frames."""
    VF,VB = motion(I1, I2)
    # Interpolate ten frames between the input images and plot them.
    return interpolate(I1, I2, VF, n, VB=VB)

def motion(I1, I2):
    # Threshold out near-zero rainfall intensities.
    I1_ = I1.copy()
    I1_[I1 < 0.05] = np.nan
    I2_ = I2.copy()
    I2_[I2 < 0.05] = np.nan
    # Convert the rainfall maps to unsigned byte, as required by the Optflow 
    # motion detection algorithms. Gaussian filter with std. dev. 3 is applied.
    I1 = utils.rainfall_to_ubyte(I1, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    I2 = utils.rainfall_to_ubyte(I2, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    # Compute the motion field by using the Python <-> C++ API and the Proesmans 
    # algorithm.
    return extract_motion_proesmans(I1, I2, lam=25.0, num_iter=250, num_levels=6)

def plot_rainmap(r):
    plt.figure()
    plt.imshow(r, vmin=0.05, vmax=10)
    cb = plt.colorbar()
    cb.set_label("rain rate (mm/h)")
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("input1.png", bbox_inches="tight")

def nc_r(ncdata):
    dbz = ncdata.variables['DBZ']
    dbzdata = dbz[0,0,:,:]
    dbzdata_corrected = dbzdata+2
    z=10.**(dbzdata_corrected/10.)
    return 0.0292*z**(0.6536)

def ensure_path(directory):
    """Make sure the path exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def ncdatetime(ncdata):
    times = ncdata.variables['time']
    return nc.num2date(times[:], times.units)
    
testpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/test'
gridpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/grid'

filename = 'ncf_20160903_070054.nc'
#filename1 = 'ncf_20160903_070311.nc'
filename0 = 'ncf_20160904_033827.nc'
filename1 = 'ncf_20160904_034311.nc'
testfilepath0 = os.path.join(testpath, filename0)
testfilepath1 = os.path.join(testpath, filename1)
nc0 = nc.Dataset(testfilepath0)
nc1 = nc.Dataset(testfilepath1)
datafilepath = os.path.join(gridpath, 'KUM', '20160903', filename)
datadir = os.path.dirname(datafilepath)
ncdata = nc.Dataset(datafilepath, 'r')
discardir = os.path.join(datadir, 'discarded')
ensure_path(discardir)

if 'KDP' not in list(ncdata.variables):
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(datafilepath))
    os.rename(datafilepath, discardfilepath)
    #continue

I1 = nc_r(nc0)
I2 = nc_r(nc1)

r = nc_r(ncdata)

interp(I1, I2)

#gdata = gio.read_radx_grid(datafilepath)