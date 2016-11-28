#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
#import netCDF4 as nc
import os
from pyoptflow import utils
from pyoptflow.core import extract_motion_proesmans
from pyoptflow.interpolation import interpolate
import matplotlib.pyplot as plt
import numpy as np
import glob
import itertools
import datetime
import scipy.io
import radx

plt.ioff()
debug = True

interval_s = 10.
interval_dt = datetime.timedelta(seconds=interval_s)

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
gridpath = os.path.join(basepath, 'grid')
intrp_path = os.path.join(basepath, 'interpolated')


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

def discard(filepath, ncdata):
    """Return True if the file was discarded."""
    datadir = os.path.dirname(filepath)
    discardir = os.path.join(datadir, 'discarded')
    radx.ensure_path(discardir)
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(filepath))
    os.rename(filepath, discardfilepath)

def batch_interpolate(filepaths_good, outpath, data_site=None, save_png = False):
    #elev = []
    #l_dt = []
    #l_t0_str = []
    #l_t1_str = []
    for f0, f1 in itertools.izip(filepaths_good, filepaths_good[1:]):
        if data_site is None:
            for site in radx.SITES:
                if site in f0:
                    data_site = site
        nc0 = radx.RADXgrid(f0)
        nc1 = radx.RADXgrid(f1)
        if data_site == 'KER':
            nc0, nc1 = radx.equalize_ker_zmin(nc0, nc1)
        #elev.append(nc0.variables['z0'][0])
        I1 = nc0.rainrate()
        I2 = nc1.rainrate()
        t0 = nc0.elevation_end_time()
        t1 = nc1.elevation_end_time()
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
            matsitepath = radx.ensure_path(os.path.join(outpath, data_site, 'R', 'mat', datedir))
            matfilepath = os.path.join(matsitepath, matfname)
            mdict = {'time': np.array(str(t)), 'R': r}
            scipy.io.savemat(matfilepath, mdict, do_compression=True)
            if save_png:
                pngfname = fbasename + '.png'
                pngsitepath = radx.ensure_path(os.path.join(outpath, data_site, 'R', 'png', datedir))
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
    #testfilepaths = glob.glob(os.path.join(testpath, 'KUM', '*.nc'))
    testfilepaths.sort()
    testfilepaths_good = radx.filter_filepaths(testfilepaths)
    testncs=[radx.RADXgrid(f) for f in testfilepaths_good]
    test_interp = False
    if test_interp:
        test_intrp_path = os.path.join(testpath, 'interpolated')
        batch_interpolate(testfilepaths_good, test_intrp_path, save_png=True)
    kumfilepath = os.path.join(testpath, 'ncf_20160904_033827.nc')
    kerfilepath = os.path.join(testpath, 'ncf_20160904_033918.nc')
    kerVOL_Afilepath = os.path.join(testpath, 'KER', '03', 'ncf_20160903_130208.nc')
    kerFMIBfilepath = os.path.join(testpath, 'KER', '03', 'ncf_20160903_130417.nc')
    vanfilepath = os.path.join(testpath, 'ncf_20160904_034033.nc')
    irmafilepath = os.path.join(testpath, 'ncf_20160903_130933.nc')
    irma = radx.RADXgrid(irmafilepath)
    kumnc = radx.RADXgrid(kumfilepath)
    kernc = radx.RADXgrid(kerfilepath)
    vannc = radx.RADXgrid(vanfilepath)
    vol_a = radx.RADXgrid(kerVOL_Afilepath, 'r')
    fmib = radx.RADXgrid(kerFMIBfilepath)
    vol_a.z_min = fmib.z_min
    for task in (vol_a, fmib):
        plt.figure()
        plt.imshow(task.dbz(), vmin=-20, vmax=60)
        plt.title('corrected DBZ for ' + task.task_name)
        plt.colorbar()
else:
    for site in radx.SITES:
        filepaths_all = glob.glob(os.path.join(gridpath, site, '*', 'ncf_20160903_[12]?????.nc'))
        filepaths_all.extend(glob.glob(os.path.join(gridpath, site, '*', 'ncf_20160904_0[0-6]????.nc')))
        filepaths_all.sort()
        filepaths_good = radx.filter_filepaths(filepaths_all)
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