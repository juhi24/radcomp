#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
#import netCDF4 as nc
import os
import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
import itertools
from glob import glob
from radcomp.qpe import radx, radxpaths, interpolation
from j24 import ensure_dir

plt.ion()
debug = True

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
if debug:
    GRIDPATH = os.path.join(basepath, 'test', 'cf', 'grids')
else:
    GRIDPATH = os.path.join(basepath, 'grids')
intrp_path = os.path.join(GRIDPATH, 'interpolated')


def discard(filepath, ncdata):
    """Return True if the file was discarded."""
    datadir = os.path.dirname(filepath)
    discardir = os.path.join(datadir, 'discarded')
    ensure_dir(discardir)
    ncdata.close()
    discardfilepath = os.path.join(discardir, os.path.basename(filepath))
    os.rename(filepath, discardfilepath)


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


def filepaths_sep3(fromlist=True, gridpath=GRIDPATH):
    if fromlist:
        fpaths_dict = radxpaths.load()
        fpaths_df = pd.concat(fpaths_dict.values())
        fpaths_good = list(fpaths_df.filepath.values)
        fpaths_good.sort()
    else:
        fpaths_all = glob(os.path.join(gridpath, site, '*', 'ncf_20160903_[12]?????.nc'))
        fpaths_all.extend(glob(os.path.join(gridpath, site, '*', 'ncf_20160904_0[0-6]????.nc')))
        fpaths_all.sort()
        fpaths_good = radxpaths.filter_filepaths(fpaths_all)
    return fpaths_good


if False:
    for site in radx.SITES:
        filepaths_good = filepaths_sep3()
        interpolation.batch_interpolate(filepaths_good, intrp_path,
                                        data_site=site, save_png=True)

