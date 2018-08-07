# coding: utf-8
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage.filters import median_filter
from radcomp.vertical import NAN_REPLACEMENT


# CONFIG
MEDIAN_WINDOW_KDP = (20, 1)
MEDIAN_WINDOW_ZDR = (5, 1)


def dict_keys_lower(d):
    """list of dictionary keys in lower case"""
    return list(map(str.lower, d.keys()))


def create_filtered_fields_if_missing(pn, keys):
    """Check that lower case key versions of the fields exist."""
    pn_new = pn.copy()
    #filtered_fields_exist = True
    keys = list(map(str.upper, keys))
    for key in keys:
        if key.lower() not in pn_new.items:
            #filtered_fields_exist = False
            pn_new[key.lower()] = pn_new[key]
    #if not filtered_fields_exist:
        #for key in keys:
            #pn_new[key.lower()] = pn_new[key]
    return pn_new


def fltr_ground_clutter_median(pn, heigth_px=35, crop_px=20, size=(22, 2)):
    """gc filter using a combination of threshold and median filter"""
    pn_new = pn.copy()
    ground_threshold = dict(ZDR=3.5, KDP=0.22)
    keys = dict_keys_lower(ground_threshold)
    pn_new = create_filtered_fields_if_missing(pn_new, keys)
    for field in keys:
        view = pn_new[field].iloc[:heigth_px]
        fltrd = median_filter_df(view, param=field, fill=True,
                                 nullmask=pn.ZH.isnull(), size=size)
        new_values = fltrd.iloc[:crop_px]
        selection = pn_new[field]>ground_threshold[field.upper()]
        selection.loc[:, selection.iloc[crop_px]] = False # not clutter
        selection.loc[:, selection.iloc[0]] = True
        selection.iloc[crop_px:] = False
        df = pn_new[field].copy()
        df[selection] = new_values[selection]
        pn_new[field] = df
    return pn_new


def fltr_median(pn, window_kdp=MEDIAN_WINDOW_KDP,
                window_zdr=MEDIAN_WINDOW_ZDR):
    """Apply median filter on ZDR and KDP."""
    pn_out = pn.copy()
    sizes = {'ZDR': window_zdr, 'KDP': window_kdp} # filter window sizes
    # filtered field names are same as originals but in lower case
    keys = dict_keys_lower(sizes)
    new = create_filtered_fields_if_missing(pn_out, sizes.keys())[keys]
    nullmask = pn['ZH'].isnull()
    for field, data in new.iteritems():
        df = median_filter_df(data, param=field, nullmask=nullmask,
                              size=sizes[field.upper()])
        pn_out[field] = df
    return pn_out


def fltr_zdr_using_rhohv(pn):
    """Filter high ZDR values where rhohv is low."""
    pn_out = create_filtered_fields_if_missing(pn, ['ZDR'])
    high_zdr = pn['ZDR'] > 3
    low_rhohv = pn['RHO'] < 0.8
    #high_zdr2 = pn['ZDR'] > 2
    #low_rhohv2 = pn['RHO'] < 0.7
    cond = (high_zdr & low_rhohv)# | (high_zdr2 & low_rhohv2)
    pn_out['zdr'][cond] = np.nan
    return pn_out


def reject_outliers(df, m=2):
    d = df.subtract(df.median(axis=1), axis=0).abs()
    mdev = d.median(axis=1)
    s = d.divide(mdev, axis=0).replace(np.inf, np.nan).fillna(0)
    return df[s<m].copy()


def fltr_ground_clutter(pn_orig, window=18, ratio_limit=8):
    """simple threshold based gc filter"""
    # deprecated?
    pn = pn_orig.copy()
    threshold = dict(ZDR=4, KDP=0.28)
    keys = dict_keys_lower(threshold)
    pn = create_filtered_fields_if_missing(pn, keys)
    for field, data in pn.iteritems():
        if field not in keys:
            continue
        for dt, col in data.iteritems():
            winsize=1
            while winsize<window:
                winsize += 1
                dat = col.iloc[:winsize].copy()
                med = dat.median()
                easy_thresh = 0.75*threshold[field.upper()]
                if med < easy_thresh or np.isnan(col.iloc[0]):
                    break # Do not filter.
                threshold_exceeded = dat.isnull().any() and med > threshold[field.upper()]
                median_limit_exceeded = med > ratio_limit*dat.abs().min()
                view = pn[field, :, dt].iloc[:window]
                if median_limit_exceeded:
                    view[view>0.95*med] = NAN_REPLACEMENT[field.upper()]
                    break
                if threshold_exceeded:
                    view[view>threshold[field.upper()]] = NAN_REPLACEMENT[field.upper()]
                    break
    return pn


def median_filter_df(df, param=None, fill=True, nullmask=None, **kws):
    """median_filter wrapper for DataFrames"""
    if nullmask is None:
        nullmask = df.isnull()
    if fill and param is not None:
        df_new = df.fillna(NAN_REPLACEMENT[param.upper()])
    else:
        df_new = df.copy()
    result = median_filter(df_new, **kws)
    try:
        result = pd.DataFrame(result, index=df_new.index, columns=df_new.columns)
    except AttributeError: # input was Series
        result = pd.DataFrame(result, index=df_new.index)
    if param is not None:
        result[result.isnull()] = NAN_REPLACEMENT[param.upper()]
    result[nullmask] = np.nan
    return result


def savgol_series(data, *args, **kws):
    """savgol filter for Series"""
    result_arr = savgol_filter(data.values.flatten(), *args, **kws)
    return pd.Series(index=data.index, data=result_arr)


def replace_values(s, cond, replacement=np.nan):
    """Replace values based on condition."""
    out = s.copy()
    out[cond] = replacement
    return out


def fltr_no_hydrometeors(s, rho, rholim=0.97, n_thresh=2):
    """Filter values where rhohv limit is not reached in the profile."""
    no_hydrometeors = (rho > rholim).sum() < n_thresh
    return replace_values(s, no_hydrometeors)


def fltr_rolling_median_thresh(s, window=6, threshold=10):
    """Filter anomalous values by checking deviation from rolling median."""
    rolling_median = s.rolling(window, center=True, min_periods=1).median()
    cond = (s-rolling_median).apply(abs) > threshold
    return replace_values(s, cond)


