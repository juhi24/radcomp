#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""

import scipy.io
import glob
from os import path
import numpy as np
import pandas as pd
from radcomp import radx
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
import getpass
import j24

debug = False
save_png = False
plt.ioff()
#plt.close('all')

home = j24.home()
basepath = path.join('/media', getpass.getuser(), '04fafa8f-c3ca-48ee-ae7f-046cf576b1ee')
resultspath = path.join(home, 'results', 'radcomp', 'interpolate')
gridpath = path.join(basepath, 'grid')
intrp_path = path.join(basepath, 'interpolated')
composite_r_path = path.join(intrp_path, 'composite', 'R')
composite_frames_path = path.join(intrp_path, 'composite', 'video_frames')
#matpath_pattern = os.path.join(intrp_path, '???', 'R', 'mat')
matpath_pattern = path.join(intrp_path, '???', 'R', 'mat_comb')

times = pd.date_range(pd.datetime(2016, 9, 3, 10, 1), pd.datetime(2016, 9, 4, 6, 59), freq='Min').tolist()

df_radx = pd.DataFrame(columns=radx.SITES, index=times)
filepaths_all = glob.glob(path.join(gridpath, '???', '*', 'ncf_20160903_[12]?????.nc'))
filepaths_all.extend(glob.glob(path.join(gridpath, '???', '*', 'ncf_20160904_0[0-6]????.nc')))
filepaths_all.sort()
filepaths_good = radx.filter_filepaths(filepaths_all) # takes a lot of time!

rs = pd.Series(name='rainrate', index=times)
sitencs_old = {'KER': None, 'KUM': None, 'VAN': None}
dbz_d = copy.deepcopy(sitencs_old)
initialize = 10
space = 0.05
if not debug:
    for dt in times:
        dt_nice = dt.strftime('%Y-%m-%d %H:%M')
        dtdir = dt.strftime('%Y%m%d')
        dtstr = dt.strftime('%Y%m%d_%H%M')
        matfname = 'intrp_' + dtstr + '.mat'
        gridfname_pattern = 'ncf_' + dtstr + '??.nc'
        sitencs = sitencs_old
        for site in radx.SITES:
            gridfpath_pattern = path.join(gridpath, site, dtdir, gridfname_pattern)
            for fp in filepaths_good:
                if glob.fnmatch.fnmatch(fp, gridfpath_pattern):
                    sitencs[site] = radx.RADXgrid(fp)
                    sitencs[site].equalize_dbz = True
                    #dbz_d[site] = sitencs[site].dbz_raw()[0,0,:,:]
                    dbz_d[site] = sitencs[site].dbz()
                    break
        sitencs_old = sitencs
        matdir = j24.ensure_dir(path.join(composite_r_path, 'mat', dtdir))
        matfpath = path.join(matdir, matfname)
        r = scipy.io.loadmat(matfpath)['R']
        if initialize > 0:
            initialize -= 1
            continue
        fig = plt.figure(figsize=(9,8))
        gs = gridspec.GridSpec(1, 2, width_ratios=(2.44, 1))
        gs.update(left=0, wspace=0.04, hspace=space, right=.92, top=0.9, bottom=0.08)
        gs_r = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], height_ratios=[35,1,2.5], hspace=space, wspace=space)
        gs_dbz = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], width_ratios=[13,1], hspace=space, wspace=0.1)
        axr = plt.subplot(gs_r[0, :])
        axsite = {}
        for i, site in enumerate(radx.SITES):
            axsite[site] = plt.subplot(gs_dbz[i, 0])
        ax_cb_r = plt.subplot(gs_r[1, :])
        ax_cb_dbz = plt.subplot(gs_dbz[:, -1])
        radx.plot_rainmap(r, fig=fig, ax=axr, cax=ax_cb_r, orientation='horizontal')
        for site in radx.SITES:
            ax = axsite[site]
            im = ax.pcolormesh(dbz_d[site], vmin=-30, vmax=60, cmap='gist_ncar')
            ax.set_title(site, y=0.82, x=0.82)
            ax.set_xticks([])
            ax.set_yticks([])
        axr.set_title(dt_nice, y=0.9768, x=0.99, va='top', ha='right', bbox={'facecolor':'white'})
        cb_dbz = fig.colorbar(im, cax=ax_cb_dbz)
        cb_dbz.set_label('reflectivity (dBZ)')
        outdir = j24.ensure_dir(path.join(composite_frames_path, dtdir))
        outpath = path.join(outdir, 'pvid_' + dtstr + '.png')
        fig.savefig(dt.strftime(outpath), bbox_inches='tight')
        plt.close(fig)