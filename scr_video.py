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
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

debug = False
save_png = False
plt.ion()
plt.close('all')

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
gridpath = os.path.join(basepath, 'grid')
intrp_path = os.path.join(basepath, 'interpolated')
composite_r_path = os.path.join(intrp_path, 'composite', 'R')
#matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat')
matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat_comb')

times = pd.date_range(pd.datetime(2016, 9, 3, 10, 01), pd.datetime(2016, 9, 4, 6, 59), freq='Min').tolist()

df_radx = pd.DataFrame(columns=radx.SITES, index=times)
filepaths_all = glob.glob(os.path.join(gridpath, '???', '*', 'ncf_20160903_[12]?????.nc'))
filepaths_all.extend(glob.glob(os.path.join(gridpath, '???', '*', 'ncf_20160904_0[0-6]????.nc')))
filepaths_all.sort()
#filepaths_good = radx.filter_filepaths(filepaths_all)

rs = pd.Series(name='rainrate', index=times)
sitencs_old = {'KER': None, 'KUM': None, 'VAN': None}
dbz_d = copy.deepcopy(sitencs_old)
initialize = 10
if not debug:
    for dt in times:
        dt_nice = dt.strftime('%Y-%m-%d %H:%M')
        dtdir = dt.strftime('%Y%m%d')
        dtstr = dt.strftime('%Y%m%d_%H%M')
        matfname = 'intrp_' + dtstr + '.mat'
        gridfname_pattern = 'ncf_' + dtstr + '??.nc'
        sitencs = sitencs_old
        for site in radx.SITES:
            gridfpath_pattern = os.path.join(gridpath, site, dtdir, gridfname_pattern)
            for fp in filepaths_good:
                if glob.fnmatch.fnmatch(fp, gridfpath_pattern):
                    sitencs[site] = radx.RADXgrid(fp)
                    dbz_d[site] = sitencs[site].dbz_raw()[0,0,:,:]
                    break
        sitencs_old = sitencs
        matfpath = os.path.join(composite_r_path, 'mat', dtdir, matfname)
        r = scipy.io.loadmat(matfpath)['R']
        if initialize > 0:
            initialize -= 1
            continue
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=(1.75, 1), left=0, wspace=0, hspace=0, right=.92, top=0.9, bottom=0.08)
        gs_r = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], height_ratios=[25,1], hspace=0.01, wspace=0.01)
        gs_dbz = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], width_ratios=[15,1], hspace=0.01, wspace=0.01)
        axr = plt.subplot(gs_r[:-1, :])
        axsite = {}
        for i, site in enumerate(radx.SITES):
            axsite[site] = plt.subplot(gs_dbz[i, 0])
        ax_cb_r = plt.subplot(gs_r[-1, :])
        ax_cb_dbz = plt.subplot(gs_dbz[:, -1])
        radx.plot_rainmap(r, fig=fig, ax=axr, cax=ax_cb_r, orientation='horizontal')
        for site in radx.SITES:
            ax = axsite[site]
            im = ax.imshow(dbz_d[site], vmin=-30, vmax=60)
            ax.set_title(site, y=0.82, x=0.82)
            ax.set_xticks([])
            ax.set_yticks([])
        axr.set_title(dt_nice, y=0.93, x=0.8)
        cb_dbz = fig.colorbar(im, cax=ax_cb_dbz, use_gridspec=False)
        cb_dbz.set_label('reflectivity (dBZ)')