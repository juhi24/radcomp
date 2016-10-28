#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import netCDF4 as nc
import os
#import pandas as pd
#import h5py
from pyoptflow import utils
from pyoptflow.core import extract_motion_proesmans
from pyoptflow.interpolation import interpolate
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy
import itertools
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

def discard(filepath, ncdata):
    """Return True if the file was discarded."""
    datadir = os.path.dirname(filepath)
    discardir = os.path.join(datadir, 'discarded')
    ensure_path(discardir)
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(filepath))
    os.rename(filepath, discardfilepath)

def data_is_bad(ncdata):
    z = ncdata.variables['z0']
    not_ppi = False # TODO
    no_kdp = 'KDP' not in list(ncdata.variables)
    high_elevation = z[0] > 1.5
    return no_kdp or high_elevation or not_ppi

def filter_filepaths(filepaths_all):
    filepaths_good = copy.deepcopy(filepaths_all)
    for filepath in filepaths_all:
        with nc.Dataset(filepath, 'r') as ncdata:
            if data_is_bad(ncdata):
                print('bad: ' + filepath)
                filepaths_good.remove(filepath)
    return filepaths_good

def work(filepaths_all):
    pass

testpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/test'
testfilepaths = glob.glob(os.path.join(testpath, '???', '*.nc'))
testfilepaths.sort()
gridpath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/grid'

d_filepaths_all = {}
for site in ['KUM', 'KER', 'VAN']:
    d_filepaths_all[site] = glob.glob(os.path.join(gridpath, site, '*', '*.nc'))

d_filepaths_good = copy.deepcopy(d_filepaths_all)
for site in d_filepaths_all:
    pass

testfilepaths_good = filter_filepaths(testfilepaths)
dt = []
for f0, f1 in itertools.izip(testfilepaths_good, testfilepaths_good[1:]):
    nc0 = nc.Dataset(f0, 'r')
    nc1 = nc.Dataset(f1, 'r')
    I1 = nc_r(nc0)
    I2 = nc_r(nc1)
    t0 = ncdatetime(nc0)[0]
    t1 = ncdatetime(nc1)[0]
    dt.append(t1-t0)
    #interpd = interp(I1, I2)

#filename0 = 'ncf_20160904_033827.nc' # KUM
#filename1 = 'ncf_20160904_033958.nc' # KUM, dt=91s
#filename0 = 'ncf_20160904_033918.nc' # KER
#filename1 = 'ncf_20160904_034208.nc' # KER, dt=170s

def testcase():
    filename0 = 'ncf_20160904_034033.nc' # VAN
    filename1 = 'ncf_20160904_034056.nc' # VAN, dt=23s
    filepath0 = os.path.join(testpath, filename0)
    filepath1 = os.path.join(testpath, filename1)
    nc0 =  nc.Dataset(filepath0, 'r')
    nc1 =  nc.Dataset(filepath1, 'r')
    
    #datafilepath = os.path.join(gridpath, 'KUM', '20160903', filename0)
    #ncdata = nc.Dataset(datafilepath, 'r')
    I1 = nc_r(nc0)
    I2 = nc_r(nc1)
    
    interpd = interp(I1, I2)
    plot_rainmap(I1)
    for rmap in interpd:
        plot_rainmap(rmap)
    plot_rainmap(I2)
    
    t0 = ncdatetime(nc0)[0]
    t1 = ncdatetime(nc1)[0]
    dt = t1-t0 
    
    #gdata = gio.read_radx_grid(datafilepath)