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
#MLI_THRESHOLD = 2


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
    mlis = ml.indicator(scaled_data.zdr, scaled_data.ZH, rho)
    c.data['MLI'] = mlis
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI'], cmap='viridis')
    #topf.plot(marker='_', linestyle='', ax=axarr[-3], color='red', label='ML top')
    #
    mlh = ml.ml_height(mlis)
    axarr[3].axhline(mlh, color='gray')
    ml_max_change = 1500
    peaksi2, peaks2 = ml.get_peaks(mlis, hlim=(mlh-ml_max_change, mlh+ml_max_change))
    plot_peaks(peaks2, ax=axarr[3])
    ml_bot, ml_top = ml.limits_peak(peaksi2, mlis.index)
    ml_top = filtering.fltr_rolling_median_thresh(ml_top, threshold=800)
    ml_top = filtering.fltr_no_hydrometeors(ml_top, rho)
    ml_top.plot(ax=axarr[2], color='red', linestyle='', marker='_')
    #mlfit = df_rolling_apply(mlis, ml_height, hlim=(mlh-1500, mlh+1500))
    #mlfit.plot(ax=axarr[2], color='olive')
    #
    i = -4
    rhocol = rho.iloc[:, i]
    mliscol = mlis.iloc[:, i]
    peakscol = peaks2.iloc[i]
    peaksicol = peaksi2.iloc[i]
    kws = {'distance': 20, 'height': 2, 'prominence': 0.3}
    pp = signal.find_peaks(mliscol, **kws)
    promcol = signal.peak_prominences(mliscol, peaksicol[0])[0]
    w = pp[1]['peak_heights']*promcol
    #
    plt.figure()
    mliscol.plot()

