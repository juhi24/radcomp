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

def interp(I1, I2, n=1):
    """Interpolate n frames."""
    VF,VB = motion(I1, I2)
    # Interpolate n frames between the input images.
    return interpolate(I1, I2, VF, n, VB=VB)

def motion(I1, I2):
    # Convert the rainfall maps to unsigned byte, as required by the Optflow 
    # motion detection algorithms. Gaussian filter with std. dev. 3 is applied.
    Iu = []
    for i, I in enumerate([I1, I2]):
        Iu.append(utils.rainfall_to_ubyte(I, R_min=0.05, R_max=10.0, filter_stddev=3.0))
    # Compute the motion field by using the Python <-> C++ API and the Proesmans 
    # algorithm.
    return extract_motion_proesmans(Iu[0], Iu[1], lam=25.0, num_iter=250, num_levels=6)

def plot_rainmap(r):
    r_ = r.copy()
    r_[r < 0.05] = np.nan
    plt.figure()
    plt.imshow(r_, vmin=0.05, vmax=10)
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

def read_testfile(filename):
    testfilepath = os.path.join(testpath, filename)
    return nc.Dataset(testfilepath, 'r')
    
testpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/test'
gridpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/grid'

filename = 'ncf_20160903_070054.nc'
filename0 = 'ncf_20160904_033827.nc'
filename1 = 'ncf_20160904_033958.nc'
nc0 = read_testfile(filename0)
nc1 = read_testfile(filename1)

datafilepath = os.path.join(gridpath, 'KUM', '20160903', filename)
datadir = os.path.dirname(datafilepath)
ncdata = nc.Dataset(datafilepath, 'r')

# Discard bad files
discardir = os.path.join(datadir, 'discarded')
ensure_path(discardir)
if 'KDP' not in list(ncdata.variables):
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(datafilepath))
    os.rename(datafilepath, discardfilepath)
    #continue

I1 = nc_r(nc0)
I2 = nc_r(nc1)

interpd = interp(I1, I2)
plot_rainmap(I1)
plot_rainmap(interpd[0])
plot_rainmap(I2)

t0 = ncdatetime(nc0)[0]
t1 = ncdatetime(nc1)[0]
dt = t1-t0

#gdata = gio.read_radx_grid(datafilepath)