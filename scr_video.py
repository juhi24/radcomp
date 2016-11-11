#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""

import scipy.io
import glob
import os
import numpy as np
import pandas as pd
import radx
import matplotlib.pyplot as plt

debug = True
save_png = False
plt.ioff()

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
gridpath = os.path.join(basepath, 'grid')
intrp_path = os.path.join(basepath, 'interpolated')
composite_r_path = os.path.join(intrp_path, 'composite', 'R')
#matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat')
matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat_comb')

times = pd.date_range(pd.datetime(2016, 9, 3, 10, 01), pd.datetime(2016, 9, 4, 6, 59), freq='Min').tolist()

df_radx = pd.DataFrame(columns=radx.SITES, index=times)
for site in radx.SITES:
    filepaths_all = glob.glob(os.path.join(gridpath, site, '*', 'ncf_20160903_[12]?????.nc'))
    filepaths_all.extend(glob.glob(os.path.join(gridpath, site, '*', 'ncf_20160904_0[0-6]????.nc')))
    filepaths_all.sort()
    filepaths_good = radx.filter_filepaths(filepaths_all)

if not debug:
    rs = pd.Series(name='rainrate', index=times)
    for dt in times:
        dtdir = dt.strftime('%Y%m%d')
        fname = dt.strftime('intrp_%Y%m%d_%H%M.mat')
        matfpath = os.path.join(matpath_pattern, dtdir, fname)
        rs[dt] = scipy.io.loadmat(matfpath)['R']


