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

def interp(I1, I2):
    # Threshold out near-zero rainfall intensities.
    I1_ = I1.copy()
    I1_[I1 < 0.05] = np.nan
    I2_ = I2.copy()
    I2_[I2 < 0.05] = np.nan
    
    # Plot the original rainfall maps.
    plt.figure()
    plt.imshow(I1_, vmin=0.05, vmax=10)
    cb = plt.colorbar()
    cb.set_label("rain rate (mm/h)")
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("input1.png", bbox_inches="tight")
    
    plt.figure()
    plt.imshow(I2_, vmin=0.05, vmax=10)
    cb = plt.colorbar()
    cb.set_label("rain rate (mm/h)")
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("input2.png", bbox_inches="tight")
    
    # Convert the rainfall maps to unsigned byte, as required by the Optflow 
    # motion detection algorithms. Gaussian filter with std. dev. 3 is applied.
    I1_ubyte = utils.rainfall_to_ubyte(I1, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    I2_ubyte = utils.rainfall_to_ubyte(I2, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    
    # Compute the motion field by using the Python <-> C++ API and the Proesmans 
    # algorithm.
    VF,VB = extract_motion_proesmans(I1_ubyte, I2_ubyte, lam=25.0, num_iter=250, 
                                     num_levels=6)
    
    # Interpolate ten frames between the input images and plot them.
    I_interp = interpolate(I1, I2, VF, 10, VB=VB)
    for i in range(len(I_interp)):
      plt.figure()
      I = I_interp[i].copy()
      I[I < 0.05] = np.nan
      plt.imshow(I, vmin=0.05, vmax=10)
      cb = plt.colorbar()
      cb.set_label("rain rate (mm/h)")
      plt.xticks([])
      plt.yticks([])
      plt.savefig("input_interp_%d.png" % (i+1), bbox_inches="tight")
      plt.close()

def motion(I1, I2):
    # Threshold out near-zero rainfall intensities.
    I1_ = I1.copy()
    I1_[I1 < 0.05] = 1e20
    I2_ = I2.copy()
    I2_[I2 < 0.05] = 1e20
    
    # Plot the original rainfall maps.
    plot_rainmap(I1_)
    plot_rainmap(I2_)
    
    # Contour plot of the first input image on top of the second one.
    visualization.plot_contour_overlay(I1, I2, 0.05, 0.05, 10.0)
    #plt.savefig("contour_overlay.png", bbox_inches="tight")
    
    # Convert the rainfall maps to unsigned byte, as required by the Optflow 
    # motion detection algorithms. Gaussian filter with std. dev. 3 is applied.
    I1 = utils.rainfall_to_ubyte(I1, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    I2 = utils.rainfall_to_ubyte(I2, R_min=0.05, R_max=10.0, filter_stddev=3.0)
    
    # Compute the motion field by using the Python <-> C++ API and the Proesmans 
    # algorithm.
    V = extract_motion_proesmans(I1, I2, lam=25.0, num_iter=250, num_levels=6)[0]
    
    # Plot the U- and V- (horizontal and vertical) components of the motion field.
    figs = visualization.plot_motion_field_components(V, sel_comp=["U", "V"])
    figs[0].savefig("motion_U.png", bbox_inches="tight")
    figs[1].savefig("motion_V.png", bbox_inches="tight")
    
    # Plot the quality map of the motion field.
    visualization.plot_motion_field_quality(V, 0, 0.0, 1.0)
    #plt.savefig("motion_quality.png", bbox_inches="tight")
    
    # Quiver plot of the motion field.
    visualization.plot_motion_quiver(V, stride=15)
    #plt.savefig("motion_quiver.png", bbox_inches="tight")
    
    # Quiver plot on top of the input image.
    visualization.plot_motion_field_overlay(I1_, V, 0.05, 10.0, stride=15, 
                                            colorbar_label="rain rate (mm/h)")
    #plt.savefig("motion_overlay.png", bbox_inches="tight", dpi=200)

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