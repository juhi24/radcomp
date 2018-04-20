# coding: utf-8
"""implementing Brandon's ML detection method"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from radcomp.vertical import case, classification, filtering

RHO_MAX = 0.975
H_MAX = 4000


def savgol_series(data, *args, **kws):
    """savgol filter for Series"""
    result_arr = signal.savgol_filter(data.values.flatten(), *args, **kws)
    return pd.Series(index=data.index, data=result_arr)


def consecutive_grouper(s):
    """Consecutive values to have same integer -> 111223333455"""
    return (s != s.shift()).cumsum()


def first_consecutive(s):
    """only first consecutive group of trues is kept"""
    grouper = consecutive_grouper(s)
    g = s.groupby(grouper)
    true_groups = g.mean()[g.mean()]
    if true_groups.empty:
        return grouper==-1
    return grouper==[true_groups.index[0]]


def expand_ml(ml, rho):
    """expand ML"""
    for t, mlc in ml.iteritems():
        ml[t] = _expand_ml_series(mlc, rho[t])
    return ml


def _expand_ml_series(ml, rho, rholim=RHO_MAX):
    """expand detected ml using rhohv"""
    grouper = consecutive_grouper(rho<rholim)
    selected = grouper[ml]
    if selected.empty:
        return ml
    iml = selected.iloc[0]
    ml_new = grouper==iml
    ml_new = _check_above_ml(ml_new, rho, grouper, iml, rholim)
    return ml_new


def _check_above_ml(ml, rho, grouper, iml, rholim):
    """check that value above expanded ml makes sense"""
    try:
        rho_above_ml = rho[grouper==iml+1].iloc[0]
    except IndexError:
        return ml==np.inf # all False
    if (rho_above_ml < rholim) or np.isnan(rho_above_ml):
        return ml==np.inf # all False
    return ml


def ml_top(ml, maxh=H_MAX, no_ml_val=np.nan):
    """extract ml top height from detected ml"""
    top = ml[::-1].idxmax()
    if no_ml_val is not None:
        top[top>maxh] = no_ml_val
    return top


def filter_ml_top(top, size=3):
    """Apply median filter to ml top height."""
    topf = filtering.median_filter_df(top.fillna(0), size=size)
    topf[np.isnan(top)] = np.nan
    return topf


def detect_ml(mli, rho, mli_thres=2, rholim=RHO_MAX):
    """Detect ML using melting layer indicator."""
    ml = mli > mli_thres
    ml[rho>0.975] = False
    ml = ml.apply(first_consecutive)
    ml = expand_ml(ml, rho)
    return ml


def ml_indicator(zdr_scaled, zh_scaled, rho):
    """Calculate ML indicator."""
    mli = (1-rho)*(zdr_scaled+1)*zh_scaled*100
    mli = mli.apply(savgol_series, args=(5, 2))
    return mli


def false_pd(df):
    """Pandas data as all False"""
    return df.astype(bool)*False


def peak_series(s, imax=None, **kws):
    ind, props = signal.find_peaks(s, **kws)
    if imax is not None:
        ind = ind[ind<imax]
    return ind, props


def peak_series_bool(s, **kws):
    ind = peak_series(s, **kws)
    out = false_pd(s)
    out.iloc[ind] = True
    return out


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
    isort = np.argsort(arr)
    cs = w[isort].cumsum()
    cutoff = w.sum()/2
    return arr[cs>=cutoff][0]


def peak_weights(peaksi):
    heights = get_peaksi_prop(peaksi, 'peak_heights')
    prom = get_peaksi_prop(peaksi, 'prominences')
    return prom*heights


def ml_height_median(peaksi, peaks):
    weights = peak_weights(peaksi)
    warr=np.concatenate(weights.values)
    parr=np.concatenate(peaks.values)
    return weighted_median(parr, warr)


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
    mli = ml_indicator(scaled_data.zdr, scaled_data.ZH, rho)
    mlis = mli.apply(savgol_series, args=(15, 3))
    ml = detect_ml(mli, rho)
    top = ml_top(ml)
    topf = filter_ml_top(top)
    c.data['MLI'] = mlis
    c.data['ML'] = ml
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI', 'ML'], cmap='viridis')
    topf.plot(marker='_', linestyle='', ax=axarr[-3], color='red', label='ML top')
    imaxh = find(mli.index, H_MAX)
    #
    kws = dict(height=2, distance=20, prominence=0.3)
    peaksi = mlis.apply(peak_series, imax=imaxh, **kws)
    peaks = peaksi.apply(lambda i: list(mli.iloc[i[0]].index))
    heights = get_peaksi_prop(peaksi, 'peak_heights')
    prom = get_peaksi_prop(peaksi, 'prominences')
    weights = prom*heights
    plot_peaks(peaks, ax=axarr[3])
    mlh = ml_height_median(peaksi, peaks)
    axarr[3].axhline(mlh, color='gray')
    #
    i = -4
    mlcol = ml.iloc[:, i]
    mlicol = mli.iloc[:, i]
    mliscol = mlis.iloc[:, i]
    peakscol = peaks.iloc[i]
    peaksicol = peaksi.iloc[i]
    pp = signal.find_peaks(mliscol, **kws)
    promcol = signal.peak_prominences(mliscol, peaksicol[0])[0]
    w = pp[1]['peak_heights']*promcol
    #
    plt.figure()
    mlicol.plot()
    mliscol.plot()

