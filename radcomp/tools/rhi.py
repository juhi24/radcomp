# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Authors: Dmitri Moisseev and Jussi Tiira
"""

from glob import glob
from os import path
from datetime import datetime

import pyart
import numpy as np
import pandas as pd
import scipy.io as sio

from radcomp.tools import db2lin, lin2db

from j24 import eprint


R_IKA_HYDE = 64450 # m
DB_SCALED_VARS = ('ZH', 'ZDR')


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


def _interp(h_target, h_orig, var, agg_fun=np.nanmedian, **kws):
    """numpy.interp wrapper with aggregation on axis 1"""
    fun = agg_fun_chooser(agg_fun, **kws)
    var_agg = fun(var, axis=1)
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
    return dict(ZH=ZH, ZDR=ZDR, KDP=KDP, RHO=RHO, DP=DP)


def scan_timestamp(radar):
    """scan timestamp in minute resolution"""
    median_ts = np.median(radar.time['data'])
    return datetime.fromtimestamp(median_ts).replace(second=0)


def filter_range(rdr_vars, r, r_poi, r_agg):
    """Discard all data that is not within a range from a distance of interest.
    """
    rvars = rdr_vars.copy()
    rmin = r_poi - r_agg
    rmax = r_poi + r_agg
    for key in ['ZH', 'ZDR', 'KDP', 'RHO']:
        var = rvars[key]
        var.set_fill_value(np.nan)
        var.mask[r<=rmin] = True
        var.mask[r>=rmax] = True
        rvars.update({key: var.filled()})
    return rvars


def height(radar, r, r_poi):
    dr = np.abs(r-r_poi)
    ix = dr.argmin(axis=1)
    return radar.gate_z['data'][range(ix.size),ix]


def vrhi2vp(radar, **kws):
    """Extract vertical profile from volume scan slice."""
    rdr_vars, hght = rhi_preprocess(radar, **kws)
    df = pd.Panel(major_axis=hght, data=rdr_vars).apply(np.nanmedian, axis=2)
    return scan_timestamp(radar), df


def rhi2vp(radar, n_hbins=None, hbins=None, agg_fun=np.nanmedian, **kws):
    hbins = hbins or np.linspace(200, 15000, n_hbins)
    """Extract vertical profile from RHI."""
    rdr_vars, hght = rhi_preprocess(radar, **kws)
    if hght is None:
        return None, None
    rvars = dict()
    for key in rdr_vars.keys():
        db_scale = key in DB_SCALED_VARS
        rvars[key] = _interp(hbins, hght, rdr_vars[key], agg_fun,
                             db_scale=db_scale)
    df = pd.DataFrame(index=hbins, data=rvars)
    return scan_timestamp(radar), df


def rhi_preprocess(radar, r_poi=R_IKA_HYDE, r_agg=1e3):
    """Process RHI data for aggregation."""
    calibration(radar, 'differential_reflectivity', 0.5)
    calibration(radar, 'reflectivity', 3)
    try: # extracting variables
        fix_elevation(radar)
        rdr_vars = extract_radar_vars(radar)
    except Exception as e:
        eprint('[extract error] {e}'.format(e=e))
        return None, None
    r = radar.gate_x['data'] # horizontal range
    rdr_vars = filter_range(rdr_vars, r, r_poi, r_agg)
    hght = height(radar, r, r_poi)
    return rdr_vars, hght


def agg_fun_chooser(agg_fun, db_scale=False):
    """aggregation of db-scale variables"""
    if (agg_fun == np.nanmedian) or not db_scale:
        return agg_fun
    return lambda x: lin_agg(x, agg_fun=agg_fun)


def mat_workflow(path_in, path_out, agg_fun=np.nanmedian,
           fname_supl='IKA_vprhi', overwrite=False, **kws):
    """Extract profiles and save as mat."""
    n_hbins = 297
    files = np.sort(glob(path.join(path_in, "*RHI_HV*.raw")))
    ObsTime  = []
    time_filename = path.basename(files[0])[0:8]
    fileOut = path.join(path_out, time_filename + '_' + fname_supl + '.mat')
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
        ts, df = rhi2vp(radar, n_hbins=n_hbins)
        if ts is None:
            continue
        tstr = path.basename(filename)[0:12]
        ObsTime.append(datetime.strptime(tstr, '%Y%m%d%H%M').isoformat())
    print(fileOut)
    vp_rhi = {'ObsTime': ObsTime, 'height': df.index.values}
    vp_rhi.update(df.to_dict(orient='list'))
    sio.savemat(fileOut, {'VP_RHI': vp_rhi})
