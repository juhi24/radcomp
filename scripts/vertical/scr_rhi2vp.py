# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Adapted from a script by Dmitri Moisseev.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pyart
import numpy as np
import scipy.io as sio
from glob import glob
from os import path
from datetime import datetime


R_IKA_HYDE = 64450 # m


def lin_agg(db, agg_fun=np.nanmean, **kws):
    """aggregate in linear space"""
    lin = db2lin(db)
    aggregated = agg_fun(lin, **kws)
    return lin2db(aggregated)


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)


def calibration(radar, field_name, addition):
    """change radar object calibration by adding a constant value"""
    field = radar.fields[field_name].copy()
    field['data'] += addition
    radar.fields.update({field_name: field})


def interp(h_target, h_orig, var, agg_fun=np.nanmedian):
    var_agg = agg_fun(var, axis=1)
    return np.interp(h_target, h_orig, var_agg)


def fix_elevation(radar):
    """Correct elevation for antenna transition."""
    if radar.elevation['data'][0] > 90.0:
       radar.elevation['data'][0] = 0.0
    if radar.elevation['data'][1] > 90.0:
       radar.elevation['data'][1] = 0.0


def extract_radar_vars(radar, recalculate_kdp=True):
    """Extract radar variables."""
    ZH  = radar.fields['reflectivity'].copy()['data']
    ZDR = radar.fields['differential_reflectivity'].copy()['data']
    RHO = radar.fields['cross_correlation_ratio'].copy()['data']
    DP = radar.fields['differential_phase'].copy()['data']
    if recalculate_kdp:
        kdp_m = pyart.retrieve.kdp_maesaka(radar)
        KDP = np.ma.masked_array(data=kdp_m[0]['data'], mask=ZH.mask)
    else:
        KDP = radar.fields['specific_differential_phase'].copy()['data']
    return dict(zh=ZH, zdr=ZDR, kdp=KDP, rho=RHO, dp=DP)


def main(pathIn, pathOut, hmax=15000, r_agg=1e3):
    """Extract profiles and save as mat."""
    file_indx = 0
    files = np.sort(glob(path.join(pathIn, "*RHI_HV*.raw")))
    nfile = len(files)
    n_hbins = 301
    zh_vp    = np.zeros([nfile, n_hbins])
    zdr_vp   = np.zeros([nfile, n_hbins])
    kdp_vp   = np.zeros([nfile, n_hbins])
    rho_vp   = np.zeros([nfile, n_hbins])
    dp_vp    = np.zeros([nfile, n_hbins])
    ObsTime  = []
    height   = np.linspace(0, hmax, n_hbins)
    for filename in files:
        try: ## reading radar data
            print(filename)
            radar = pyart.io.read(filename)
            calibration(radar, 'differential_reflectivity', 0.5)
            calibration(radar, 'reflectivity', 3)
            fix_elevation(radar)
        except:
            print("could not read data")
            continue
        rdr_vars = extract_radar_vars(radar)

        r = radar.gate_x['data'] # horizontal range
        r_hyde = R_IKA_HYDE
        rmin = r_hyde - r_agg
        rmax = r_hyde + r_agg
        for key in ['zh', 'zdr', 'kdp', 'rho']:
            var = rdr_vars[key]
            var.set_fill_value(np.nan)
            var.mask[r<=rmin] = True
            var.mask[r>=rmax] = True
            rdr_vars.update({key: var.filled()})

        diff_r = np.abs(r[0,:] - r_hyde)
        indx = diff_r.argmin()
        hght = radar.gate_z['data'][:, indx]
        #dr = np.abs(r-r_hyde)
        #ix = dr.argmin(axis=1)
        #hght = radar.gate_z['data'][range(ix.size),ix]

        agg_fun = np.nanmedian
        def agg_fun_db(x, **kws):
            """aggregation of db-scale variables"""
            if agg_fun == np.nanmedian:
                return agg_fun(x, **kws)
            return lin_agg(x, agg_fun=agg_fun, **kws)
        zh_vp[file_indx,:]  = interp(height, hght, rdr_vars['zh'], agg_fun_db)
        zdr_vp[file_indx,:] = interp(height, hght, rdr_vars['zdr'], agg_fun_db)
        kdp_vp[file_indx,:] = interp(height, hght, rdr_vars['kdp'], agg_fun)
        rho_vp[file_indx,:] = interp(height, hght, rdr_vars['rho'], agg_fun)
        dp_vp[file_indx,:]  = interp(height, hght, rdr_vars['dp'], agg_fun)

        tstr = path.basename(filename)[0:12]
        ObsTime.append(datetime.strptime(tstr, '%Y%m%d%H%M').isoformat())
        file_indx += 1
    fname_supl = 'IKA_vprhi_1km_median_m'
    time_filename = path.basename(filename)[0:8]
    fileOut = path.join(pathOut, time_filename + '_' + fname_supl + '.mat')
    print(fileOut)
    VP_RHI = {'ObsTime': ObsTime, 'ZH': zh_vp, 'ZDR': zdr_vp, 'KDP': kdp_vp, 'RHO': rho_vp, 'DP': dp_vp, 'height': height}
    sio.savemat(fileOut, {'VP_RHI':VP_RHI})


if __name__ == '__main__':
    #pathOut = '/Users/moiseev/Data/VP_RHI/'
    #pathIn = "/Volumes/uhradar/IKA_final/20140318/"
    home = path.expanduser('~')
    pathIn = path.join(home, 'DATA', 'IKA', '20180818RHI')
    pathOut = path.join(home, 'results', 'radcomp', 'vertical', 'vp_dmitri')
    main(pathIn, pathOut)

