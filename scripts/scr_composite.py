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

save_png = False
plt.ioff()

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
intrp_path = os.path.join(basepath, 'interpolated')
composite_r_path = os.path.join(intrp_path, 'composite', 'R')
#matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat')
matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat_comb')

def composite_max(maps):
    return np.maximum.reduce(maps)

times = pd.date_range(pd.datetime(2016, 9, 3, 10, 01), pd.datetime(2016, 9, 4, 6, 59), freq='Min').tolist()

for dt in times:
    dtdir = dt.strftime('%Y%m%d')
    fname_prefix = dt.strftime('intrp_%Y%m%d_%H%M')
    fname_pattern = fname_prefix + '??.mat'
    pathsglob = os.path.join(matpath_pattern, dtdir, fname_pattern)
    mats = glob.glob(pathsglob)
    print(mats)
    rs = [scipy.io.loadmat(fpath)['R'] for fpath in mats]
    comp = composite_max(rs)
    savedict = {'R': comp, 'time': str(dt)}
    mat_outdir_path = radx.ensure_path(os.path.join(composite_r_path, 'mat', dtdir))
    mat_out_fpath = os.path.join(mat_outdir_path, fname_prefix + '.mat')
    scipy.io.savemat(mat_out_fpath, savedict)
    if save_png:
        pngfname = fname_prefix + '.png'
        pngsitepath = radx.ensure_path(os.path.join(composite_r_path, 'png', dtdir))
        pngfilepath = os.path.join(pngsitepath, pngfname)
        fig, ax = radx.plot_rainmap(comp)
        ax.set_title(str(dt))
        fig.savefig(pngfilepath, bbox_inches="tight")
        plt.close(fig)
