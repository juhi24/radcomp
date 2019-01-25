# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Authors: Dmitri Moisseev and Jussi Tiira
"""

import pyart
import numpy as np
import scipy.io as sio
from glob import glob
from os import path
from datetime import datetime
from radcomp.tools import db2lin, lin2db
from j24 import eprint


R_IKA_HYDE = 64450 # m


def lin_agg(db, agg_fun=np.nanmean, **kws):
    """aggregate in linear space"""
    lin = db2lin(db)
    aggregated = agg_fun(lin, **kws)
    return lin2db(aggregated)


def calibration(radar, field_name, addition):
    """change radar object calibration by adding a constant value"""
    field = radar.fields[field_name].copy()
    field['data'] += addition
    radar.fields.update({field_name: field})


def _interp(h_target, h_orig, var, agg_fun=np.nanmedian):
    """numpy.interp wrapper with aggregation on axis 1"""
    var_agg = agg_fun(var, axis=1)
    return np.interp(h_target, h_orig, var_agg)


def fix_elevation(radar):
    """Correct elevation for antenna transition."""
    for i in [0, 1]:
        if radar.elevation['data'][i] > 90.0:
           radar.elevation['data'][i] = 0.0


def extract_radar_vars(radar, recalculate_kdp=True):
    """Extract radar variables."""
    ZH  = radar.fields['reflectivity'].copy()['data']
    ZDR = radar.fields['differential_reflectivity'].copy()['data']
    RHO = radar.fields['cross_correlation_ratio'].copy()['data']
    DP = radar.fields['differential_phase'].copy()['data']
    if recalculate_kdp:
        try:
            kdp_m = pyart.retrieve.kdp_maesaka(radar)
        except IndexError:
            # outlier checking sometimes causes trouble (with weak kdp?)
            kdp_m = pyart.retrieve.kdp_maesaka(radar, check_outliers=False)
        KDP = np.ma.masked_array(data=kdp_m[0]['data'], mask=ZH.mask)
    else:
        KDP = radar.fields['specific_differential_phase'].copy()['data']
    return dict(zh=ZH, zdr=ZDR, kdp=KDP, rho=RHO, dp=DP)


def rhi2vp(pathIn, pathOut, hbins=None, agg_fun=np.nanmedian, r_agg=1e3,
           fname_supl='IKA_vprhi', r_hyde=R_IKA_HYDE, overwrite=False):
    """Extract profiles and save as mat."""
    n_hbins = 297
    hbins = hbins or np.linspace(200, 15000, n_hbins)
    files = np.sort(glob(path.join(pathIn, "*RHI_HV*.raw")))
    nfile = len(files)
    init = np.full([nfile, n_hbins], np.nan)
    zh_vp, zdr_vp, kdp_vp, rho_vp, dp_vp = (init.copy() for i in range(5))
    ObsTime  = []
    time_filename = path.basename(files[0])[0:8]
    fileOut = path.join(pathOut, time_filename + '_' + fname_supl + '.mat')
    if path.exists(fileOut) and not overwrite:
        print('{} [notice] file already exists, skipping.'.format(fileOut))
        return
    for file_indx, filename in enumerate(files):
        print(filename)
        try: # reading radar data
            radar = pyart.io.read(filename)
        except Exception as e:
            eprint('{fname} [read error] {e}'.format(fname=filename, e=e))
            continue
        calibration(radar, 'differential_reflectivity', 0.5)
        calibration(radar, 'reflectivity', 3)
        try: # extracting variables
            fix_elevation(radar)
            rdr_vars = extract_radar_vars(radar)
        except Exception as e:
            eprint('{fname} [extract error] {e}'.format(fname=filename, e=e))
            continue
        r = radar.gate_x['data'] # horizontal range
        rmin = r_hyde - r_agg
        rmax = r_hyde + r_agg
        for key in ['zh', 'zdr', 'kdp', 'rho']:
            var = rdr_vars[key]
            var.set_fill_value(np.nan)
            var.mask[r<=rmin] = True
            var.mask[r>=rmax] = True
            rdr_vars.update({key: var.filled()})
        #diff_r = np.abs(r[0,:] - r_hyde)
        #indx = diff_r.argmin()
        #hght = radar.gate_z['data'][:, indx]
        dr = np.abs(r-r_hyde)
        ix = dr.argmin(axis=1)
        hght = radar.gate_z['data'][range(ix.size),ix]

        def agg_fun_db(x, **kws):
            """aggregation of db-scale variables"""
            if agg_fun == np.nanmedian:
                return agg_fun(x, **kws)
            return lin_agg(x, agg_fun=agg_fun, **kws)
        zh_vp[file_indx,:]  = _interp(hbins, hght, rdr_vars['zh'], agg_fun_db)
        zdr_vp[file_indx,:] = _interp(hbins, hght, rdr_vars['zdr'], agg_fun_db)
        kdp_vp[file_indx,:] = _interp(hbins, hght, rdr_vars['kdp'], agg_fun)
        rho_vp[file_indx,:] = _interp(hbins, hght, rdr_vars['rho'], agg_fun)
        dp_vp[file_indx,:]  = _interp(hbins, hght, rdr_vars['dp'], agg_fun)

        tstr = path.basename(filename)[0:12]
        ObsTime.append(datetime.strptime(tstr, '%Y%m%d%H%M').isoformat())
    print(fileOut)
    VP_RHI = {'ObsTime': ObsTime, 'ZH': zh_vp, 'ZDR': zdr_vp, 'KDP': kdp_vp,
              'RHO': rho_vp, 'DP': dp_vp, 'height': hbins}
    sio.savemat(fileOut, {'VP_RHI':VP_RHI})
