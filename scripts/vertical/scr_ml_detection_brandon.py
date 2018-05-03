# coding: utf-8
"""implementing Brandon's ML detection method"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from radcomp.vertical import case, filtering, ml, NAN_REPLACEMENT


def filter_ml_top(top, size=3):
    """Apply median filter to ml top height."""
    topf = filtering.median_filter_df(top.fillna(0), size=size)
    topf[np.isnan(top)] = np.nan
    return topf


def plot_peaks(peaks, ax=None, **kws):
    ax = ax or plt.gca()
    for ts, vals in peaks.iteritems():
        x = np.full(len(vals), ts)
        ax.scatter(x, vals, marker='+', zorder=1, color='red')


def df_rolling_apply(df, func, w=10, **kws):
    """rolling apply on DataFrame columns"""
    size = df.shape[1]
    out = pd.Series(index=df.columns)
    for i in range(size-w):
        sli = df.iloc[:, i:i+w]
        out[int(i+w/2)] = func(sli, **kws)
    return out


if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    cases = case.read_cases('melting-test')
    c = cases.case.iloc[4]
    #
    scaled_data = case.scale_data(c.data)
    rho = c.data.RHO
    mli = ml.indicator(scaled_data.zdr, scaled_data.ZH, rho)
    c.data['MLI'] = mli
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI'], cmap='viridis')
    #
    mlh = ml.ml_height(mli)
    ml_max_change = 1500
    peaksi, peaks = ml.get_peaks(mli, hlim=(mlh-ml_max_change, mlh+ml_max_change))
    plot_peaks(peaks, ax=axarr[3])
    ml_bot, ml_top = ml.ml_limits(mli, rho)
    top = ml_top.interpolate().dropna()
    mask = ml.hseries2mask(top, mli.index)
    z_masked = c.data.ZH.fillna(NAN_REPLACEMENT['ZH'])[mask]
    zh0 = z_masked.apply(ml.collapse)
    top.plot(ax=axarr[2], color='gray', linestyle='', marker='_')
    ml_top.plot(ax=axarr[2], color='black', linestyle='', marker='_')
    #
    axarr[3].axhline(mlh, color='gray')

