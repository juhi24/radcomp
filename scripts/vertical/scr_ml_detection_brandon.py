# coding: utf-8
"""implementing Brandon's ML detection method"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from radcomp.vertical import case, classification, filtering, ml

RHO_LIM = 0.97
H_MAX = 4000
#MLI_THRESHOLD = 2


def first_or_nan(l):
    """Get first item in iterable if exists, else nan"""
    try:
        return l[0]
    except IndexError:
        return np.nan


def value_at(ind, values):
    """round index and return corresponding value, nan on ValueError"""
    try:
        return values[round(ind)]
    except ValueError:
        return np.nan


def ml_limits_peak(peaksi, heights):
    """extract ML top height from MLI peaks"""
    edges = []
    for ips_label in ('left_ips', 'right_ips'):
        ips = get_peaksi_prop(peaksi, ips_label)
        edges.append(ips.apply(first_or_nan).apply(value_at, args=(heights,)))
    return tuple(edges)


def filter_ml_top(top, size=3):
    """Apply median filter to ml top height."""
    topf = filtering.median_filter_df(top.fillna(0), size=size)
    topf[np.isnan(top)] = np.nan
    return topf


def peak_series(s, ilim=(None, None), **kws):
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


def plot_peaks(peaks, ax=None, **kws):
    ax = ax or plt.gca()
    for ts, vals in peaks.iteritems():
        x = np.full(len(vals), ts)
        ax.scatter(x, vals, marker='+', zorder=1, color='red')


def get_peaksi_prop(peaksi, prop):
    """return property from peaksi series"""
    return peaksi.apply(lambda x: x[1][prop])


def find(arr, value):
    """find closest value using argmin"""
    return abs(arr-value).argmin()


def weighted_median(arr, w):
    """general weighted median"""
    isort = np.argsort(arr)
    cs = w[isort].cumsum()
    cutoff = w.sum()/2
    try:
        return arr[isort][cs >= cutoff][0]
    except IndexError:
        return np.nan


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


def get_peaks(mlis, hlim=(0, H_MAX), height=2, width=0, distance=20,
              prominence=0.3):
    """Apply peak detection to ML indicator."""
    limits = [find(mlis.index, lim) for lim in hlim]
    peaksi = mlis.apply(peak_series, ilim=limits, height=height, width=width,
                        distance=distance, prominence=prominence)
    return peaksi, peaksi.apply(lambda i: list(mlis.iloc[i[0]].index))


def ml_height(mlis, **kws):
    """weighted median ML height from ML indicator using peak detection"""
    peaksi, peaks = get_peaks(mlis, **kws)
    return ml_height_median(peaksi, peaks)


def df_rolling_apply(df, func, w=10, **kws):
    """rolling apply on DataFrame columns"""
    size = df.shape[1]
    out = pd.Series(index=df.columns)
    for i in range(size-w):
        sli = df.iloc[:, i:i+w]
        out[int(i+w/2)] = func(sli, **kws)
    return out


basename = 'melting-test'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 5
n_clusters = 5
reduced = True
use_temperature = False
t_weight_factor = 0.8

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            n_clusters=n_clusters,
                            reduced=reduced, t_weight_factor=t_weight_factor,
                            basename=basename, use_temperature=use_temperature)

if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    cases = case.read_cases('melting-test')
    c = cases.case.iloc[4]
    c.class_scheme = scheme
    c.train()
    #
    scaled_data = case.scale_data(c.data)
    rho = c.data.RHO
    mli = ml.indicator(scaled_data.zdr, scaled_data.ZH, rho)
    mlis = mli
    c.data['MLI'] = mlis
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI'], cmap='viridis')
    #topf.plot(marker='_', linestyle='', ax=axarr[-3], color='red', label='ML top')
    imaxh = find(mli.index, H_MAX)
    #
    peaksi, peaks = get_peaks(mlis)
    heights = get_peaksi_prop(peaksi, 'peak_heights')
    prom = get_peaksi_prop(peaksi, 'prominences')
    weights = prom*heights
    mlh = ml_height_median(peaksi, peaks)
    axarr[3].axhline(mlh, color='gray')
    ml_max_change = 1500
    peaksi2, peaks2 = get_peaks(mlis, hlim=(mlh-ml_max_change, mlh+ml_max_change))
    plot_peaks(peaks2, ax=axarr[3])
    ml_bot, ml_top = ml_limits_peak(peaksi2, mlis.index)
    ml_top = filtering.fltr_rolling_median_thresh(ml_top, threshold=800)
    ml_top = filtering.fltr_no_hydrometeors(ml_top, rho)
    ml_top.plot(ax=axarr[2], color='red', linestyle='', marker='_')
    #mlfit = df_rolling_apply(mlis, ml_height, hlim=(mlh-1500, mlh+1500))
    #mlfit.plot(ax=axarr[2], color='olive')
    #
    i = -4
    rhocol = rho.iloc[:, i]
    mlicol = mli.iloc[:, i]
    mliscol = mlis.iloc[:, i]
    peakscol = peaks2.iloc[i]
    peaksicol = peaksi2.iloc[i]
    kws = {'distance': 20, 'height': 2, 'prominence': 0.3}
    pp = signal.find_peaks(mliscol, **kws)
    promcol = signal.peak_prominences(mliscol, peaksicol[0])[0]
    w = pp[1]['peak_heights']*promcol
    #
    plt.figure()
    mlicol.plot()
    mliscol.plot()

