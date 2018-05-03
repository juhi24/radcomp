# coding: utf-8
"""melting layer detection using scipy peak utilities"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
from scipy import signal
from j24.tools import find
from j24.math import weighted_median
from radcomp.vertical import filtering


H_MAX = 4000


def indicator(zdr_scaled, zh_scaled, rho, savgol_args=(15, 3)):
    """Calculate ML indicator."""
    mli = (1-rho)*(zdr_scaled+1)*zh_scaled*100
    mli = mli.apply(filtering.savgol_series, args=savgol_args)
    return mli


def get_peaksi_prop(peaksi, prop):
    """return property from peaksi series"""
    return peaksi.apply(lambda x: x[1][prop])


def peak_weights(peaksi):
    """calculate peak weights as prominence*peak_height"""
    heights = get_peaksi_prop(peaksi, 'peak_heights')
    prom = get_peaksi_prop(peaksi, 'prominences')
    return prom*heights


def ml_height_median(peaksi, peaks):
    """weighted median ML height from peak data"""
    weights = peak_weights(peaksi)
    warr = np.concatenate(weights.values)
    parr = np.concatenate(peaks.values)
    return weighted_median(parr, warr)


def peak_series(s, ilim=(None, None), **kws):
    """scipy peak detection for Series objects"""
    ind, props = signal.find_peaks(s, **kws)
    imin, imax = ilim
    up_sel, low_sel = tuple(np.ones(ind.shape).astype(bool) for i in range(2))
    if imin is not None:
        low_sel = ind > imin
    if imax is not None:
        up_sel = ind < imax
    selection = up_sel & low_sel
    for key in props:
        props[key] = props[key][selection]
    return ind[selection], props


def get_peaks(mli, hlim=(0, H_MAX), height=2, width=0, distance=20,
              prominence=0.3):
    """Apply peak detection to ML indicator."""
    limits = [find(mli.index, lim) for lim in hlim]
    peaksi = mli.apply(peak_series, ilim=limits, height=height, width=width,
                        distance=distance, prominence=prominence)
    return peaksi, peaksi.apply(lambda i: list(mli.iloc[i[0]].index))


def ml_height(mlis, **kws):
    """weighted median ML height from ML indicator using peak detection"""
    peaksi, peaks = get_peaks(mlis, **kws)
    return ml_height_median(peaksi, peaks)


def _first_or_nan(l):
    """Get first item in iterable if exists, else nan"""
    try:
        return l[0]
    except IndexError:
        return np.nan


def _value_at(ind, values):
    """round index and return corresponding value, nan on ValueError"""
    try:
        return values[round(ind)]
    except ValueError:
        return np.nan


def limits_peak(peaksi, heights):
    """ML height range from MLI peaks"""
    edges = []
    for ips_label in ('left_ips', 'right_ips'):
        ips = get_peaksi_prop(peaksi, ips_label)
        edges.append(ips.apply(_first_or_nan).apply(_value_at, args=(heights,)))
    return tuple(edges)


def ml_limits_raw(mli, ml_max_change=1500):
    """ML height range from ML indicator"""
    mlh = ml_height(mli)
    peaksi, peaks = get_peaks(mli, hlim=(mlh-ml_max_change, mlh+ml_max_change))
    return limits_peak(peaksi, mli.index)


def fltr_ml_limits(limits, rho):
    """filter ml range"""
    lims = []
    for lim in limits:
        tmp = filtering.fltr_rolling_median_thresh(lim, threshold=800)
        lims.append(filtering.fltr_no_hydrometeors(tmp, rho))
    return lims


def ml_limits(mli, rho, **kws):
    """filtered ml bottom and top heights"""
    lims = ml_limits_raw(mli, **kws)
    return fltr_ml_limits(lims, rho)


def hseries2mask(hseries, hindex):
    """boolean mask DataFrame with False below given height limits"""
    return hseries.apply(lambda x: pd.Series(data=hindex>x, index=hindex)).T


def collapse(s_filled_masked):
    """reset ground level according to mask"""
    return s_filled_masked.shift(-s_filled_masked.isnull().sum())
