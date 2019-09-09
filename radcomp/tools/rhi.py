# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Authors: Dmitri Moisseev and Jussi Tiira
"""

from glob import glob
from os import path
from datetime import datetime, timedelta

import pyart
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from radcomp.tools import db2lin, lin2db

from j24 import eprint


R_IKA_HYDE = 64450 # m
AZIM_IKA_HYDE = 81.89208 # deg
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


def extract_radar_vars(radar, recalculate_kdp=True, **kws):
    """Extract radar variables."""
    ZH  = radar.fields['reflectivity'].copy()['data']
    ZDR = radar.fields['differential_reflectivity'].copy()['data']
    RHO = radar.fields['cross_correlation_ratio'].copy()['data']
    DP = radar.fields['differential_phase'].copy()['data']
    if recalculate_kdp:
        try:
            kdp_m = pyart.retrieve.kdp_maesaka(radar, **kws)
        except IndexError:
            # outlier checking sometimes causes trouble (with weak kdp?)
            eprint('Skipping outlier check.')
            kdp_m = pyart.retrieve.kdp_maesaka(radar, check_outliers=False,
                                               **kws)
        KDP = np.ma.masked_array(data=kdp_m[0]['data'], mask=DP.mask)
    else:
        KDP = radar.fields['specific_differential_phase'].copy()['data']
    return dict(ZH=ZH, ZDR=ZDR, KDP=KDP, RHO=RHO, DP=DP)


def scan_timestamp(radar):
    """scan timestamp in minute resolution"""
    t_start = radar.time['units'].split(' ')[-1]
    median_ts = np.median(radar.time['data'])
    t = pd.to_datetime(t_start) + timedelta(seconds=median_ts)
    return t.replace(second=0, microsecond=0).to_datetime64()


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


def plot_compare_kdp(vrhi):
    mask = vrhi.fields['differential_phase']['data'].mask
    kdp = pyart.retrieve.kdp_maesaka(vrhi, Clpf=5000)[0]
    kdp['data'] = np.ma.masked_array(data=kdp['data'], mask=mask)
    vrhi.fields['kdp'] = kdp
    fig, axarr = plt.subplots(ncols=2, figsize=(12,5))
    disp = pyart.graph.RadarDisplay(vrhi)
    disp.plot('specific_differential_phase', vmin=0, vmax=0.3, ax=axarr[0], cmap='viridis')
    disp.plot('kdp', vmin=0, vmax=0.3, ax=axarr[1])
    return disp, axarr


def vrhi2vp(radar, h_thresh=60, Clpf=5000, **kws):
    """Extract vertical profile from volume scan slice."""
    #plot_compare_kdp(radar)
    rdr_vars, hght = rhi_preprocess(radar, Clpf=Clpf, **kws)
    # TODO: Panel
    df = pd.Panel(major_axis=hght, data=rdr_vars).apply(np.nanmedian, axis=2)
    df.index.name = 'height'
    h = np.array([580, 1010, 1950, 3650, 5950, 10550])
    h_norm = np.linalg.norm(df.index.values-h)
    if h_norm > h_thresh:
        efmt = 'Altitudes do not match preset values: {} > {}'
        raise ValueError(efmt.format(h_norm, h_thresh))
    df.index = h
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


def rhi_preprocess(radar, r_poi=R_IKA_HYDE, r_agg=1e3, **kws):
    """Process RHI data for aggregation."""
    calibration(radar, 'differential_reflectivity', 0.5)
    calibration(radar, 'reflectivity', 3)
    try: # extracting variables
        fix_elevation(radar)
        rdr_vars = extract_radar_vars(radar, **kws)
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


def create_volume_scan(files):
    """volume scan from multiple radar data files"""
    r = None
    for f in files:
        if r is None:
            r = pyart.io.read(f)
            continue
        r = pyart.util.join_radar(r, pyart.io.read(f))
    return r


def xarray_workflow(dir_in, dir_out=None, **kws):
    """Extract profiles from volume scans as xarray Dataset."""
    fnames = pd.Series(glob(path.join(dir_in, '*PPI3_[A-F].raw')))
    tstrs = fnames.apply(lambda s: s[-27:-15])
    g = fnames.groupby(tstrs)
    vrhis = dict()
    for tstr, df in g:
        print(tstr)
        df.sort_values(inplace=True)
        vs = create_volume_scan(df)
        vrhi = pyart.util.cross_section_ppi(vs, [AZIM_IKA_HYDE])
        t, vp = vrhi2vp(vrhi, **kws)
        vrhis[t] = vp
    df = pd.concat(vrhis)
    df.index.rename(['time', 'height'], inplace=True)
    ds = df.to_xarray()
    if dir_out is not None:
        t = ds.time.values[0]
        fname = pd.to_datetime(t).strftime('%Y%m%d_IKA_vpvol.nc')
        ds.to_netcdf(path.join(dir_out, fname))
    return ds


def mat_workflow(dir_in, dir_out, fname_supl='IKA_vprhi', overwrite=False,
                 **kws):
    """Extract profiles and save as mat."""
    n_hbins = 297
    files = np.sort(glob(path.join(dir_in, "*RHI_HV*.raw")))
    ObsTime  = []
    time_filename = path.basename(files[0])[0:8]
    fileOut = path.join(dir_out, time_filename + '_' + fname_supl + '.mat')
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
