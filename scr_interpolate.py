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
import datetime
import scipy.io
import radx
#import pyart
#import grid_io_withradx2gridread as gio

plt.ioff()
debug = True

interval_s = 10.
interval_dt = datetime.timedelta(seconds=interval_s)

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
gridpath = os.path.join(basepath, 'grid')
intrp_path = os.path.join(basepath, 'interpolated')

SITES = ['KUM', 'KER', 'VAN']
#MATPATH = os.path.join(intrp_path, 'mat')
#PNGPATH = os.path.join(intrp_path, 'png')


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

def ensure_path(directory):
    """Make sure the path exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

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
    high_elev = z[0] > 1.5
    low_elev = z[0] < 0.05
    is_correct_van_elev = int(round(ncdata.variables['z0'][0]*10)) == 7
    van_wrong_elev = ncdata.title == 'VANTAA' and not is_correct_van_elev
    return no_kdp or high_elev or not_ppi or van_wrong_elev or low_elev

def filter_filepaths(filepaths_all):
    filepaths_good = copy.deepcopy(filepaths_all)
    for filepath in filepaths_all:
        with nc.Dataset(filepath, 'r') as ncdata:
            if data_is_bad(ncdata):
                #print('bad: ' + filepath)
                filepaths_good.remove(filepath)
    return filepaths_good

def batch_interpolate(filepaths_good, outpath, data_site=None, save_png = False):
    #elev = []
    #l_dt = []
    #l_t0_str = []
    #l_t1_str = []
    for f0, f1 in itertools.izip(filepaths_good, filepaths_good[1:]):
        if data_site is None:
            for site in SITES:
                if site in f0:
                    data_site = site
        nc0 = radx.RADXgrid(f0)
        nc1 = radx.RADXgrid(f1)
        #elev.append(nc0.variables['z0'][0])
        I1 = nc0.rainrate()
        I2 = nc1.rainrate()
        t0 = nc0.datetime()[0]
        t1 = nc1.datetime()[0]
        dt = t1-t0
        #l_t0_str.append(str(t0))
        #l_t1_str.append(str(t1))
        n = int(round(dt.total_seconds()/interval_s))
        intrp_timestamps = [t0+i*interval_dt for i in range(1, n+1, 1)]
        selection = [t.second<10 for t in intrp_timestamps]
        #l_dt.append(dt)
        print(dt.total_seconds())
        intrp = np.array(interp(I1, I2, n))
        for i in np.where(selection)[0]:
            t = intrp_timestamps[i]
            r = intrp[i]
            datedir = t.strftime('%Y%m%d')
            fbasename = t.strftime('intrp_%Y%m%d_%H%M%S')
            matfname = fbasename + '.mat'
            matsitepath = ensure_path(os.path.join(outpath, data_site, 'R', 'mat', datedir))
            matfilepath = os.path.join(matsitepath, matfname)
            mdict = {'time': np.array(str(t)), 'R': r}
            scipy.io.savemat(matfilepath, mdict, do_compression=True)
            if save_png:
                pngfname = fbasename + '.png'
                pngsitepath = ensure_path(os.path.join(outpath, data_site, 'R', 'png', datedir))
                pngfilepath = os.path.join(pngsitepath, pngfname)
                fig, ax = radx.plot_rainmap(r)
                ax.set_title(str(t))
                fig.savefig(pngfilepath, bbox_inches="tight")
                plt.close(fig)


def dts(filepaths):
    l_dt = []
    for f0, f1 in itertools.izip(filepaths, filepaths[1:]):
        nc0 = radx.RADXgrid(f0)
        nc1 = radx.RADXgrid(f1)
        t0 = nc0.datetime()[0]
        t1 = nc1.datetime()[0]
        dt = t1-t0
        l_dt.append(dt)
    return l_dt


if debug:
    testpath = os.path.join(basepath, 'test')
    testfilepaths = glob.glob(os.path.join(testpath, 'KER', '03', '*.nc'))
    testfilepaths.sort()
    testfilepaths_good = filter_filepaths(testfilepaths)
    test_interp = True
    if test_interp:
        test_intrp_path = os.path.join(testpath, 'interpolated')
        batch_interpolate(testfilepaths_good, test_intrp_path, save_png=True)
    kumfilepath = os.path.join(testpath, 'ncf_20160904_033827.nc')
    kerfilepath = os.path.join(testpath, 'ncf_20160904_033918.nc')
    kerVOL_Afilepath = os.path.join(testpath, 'KER', '03', 'ncf_20160903_130208.nc')
    kerFMIBfilepath = os.path.join(testpath, 'KER', '03', 'ncf_20160903_130417.nc')
    vanfilepath = os.path.join(testpath, 'ncf_20160904_034033.nc')
    kumnc = radx.RADXgrid(kumfilepath)
    kernc = radx.RADXgrid(kerfilepath)
    vannc = radx.RADXgrid(vanfilepath)
    vol_a = radx.RADXgrid(kerVOL_Afilepath, 'r')
    fmib = radx.RADXgrid(kerFMIBfilepath)
    zmin = vol_a.z_min()
else:
    for site in SITES:
        filepaths_all = glob.glob(os.path.join(gridpath, site, '*', '*.nc'))
        filepaths_all.sort()
        filepaths_good = filter_filepaths(filepaths_all)
        batch_interpolate(filepaths_good, intrp_path, data_site=site, save_png=True)

def testcase():
    #filename0 = 'ncf_20160904_033827.nc' # KUM
    #filename1 = 'ncf_20160904_033958.nc' # KUM, dt=91s
    #filename0 = 'ncf_20160904_033918.nc' # KER
    #filename1 = 'ncf_20160904_034208.nc' # KER, dt=170s
    filename0 = 'ncf_20160904_034033.nc' # VAN
    filename1 = 'ncf_20160904_034056.nc' # VAN, dt=23s
    filepath0 = os.path.join(testpath, filename0)
    filepath1 = os.path.join(testpath, filename1)
    nc0 =  radx.RADXgrid(filepath0)
    nc1 =  radx.RADXgrid(filepath1)
    
    #datafilepath = os.path.join(gridpath, 'KUM', '20160903', filename0)
    #ncdata = nc.Dataset(datafilepath, 'r')
    I1 = nc0.rainrate()
    I2 = nc1.rainrate()
    
    interpd = interp(I1, I2)
    radx.plot_rainmap(I1)
    for rmap in interpd:
        radx.plot_rainmap(rmap)
    radx.plot_rainmap(I2)
    
    t0 = nc0.datetime()[0]
    t1 = nc1.datetime()[0]
    dt = t1-t0 
    
    #gdata = gio.read_radx_grid(datafilepath)