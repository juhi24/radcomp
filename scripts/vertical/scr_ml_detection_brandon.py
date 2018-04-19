# coding: utf-8
"""implementing Brandon's ML detection method"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot
from scipy import ndimage, signal
from radcomp.vertical import case, classification, filtering

RHO_LIMIT = 0.975


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


def _expand_ml_series(ml, rho, rholim=RHO_LIMIT):
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


def ml_top(ml, maxh=4500, no_ml_val=np.nan):
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


def detect_ml(mli, rho, mli_thres=2, rholim=RHO_LIMIT):
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


def peak_series(s, thres=0.8, min_dist=5):
    return list(peakutils.indexes(s.fillna(0), thres=thres, min_dist=min_dist))


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
    c = cases.case.iloc[3]
    c.class_scheme = scheme
    c.train()
    #
    scaled_data = case.scale_data(c.data)
    rho = c.data.RHO
    mli = ml_indicator(scaled_data.zdr, scaled_data.ZH, rho)
    ml = detect_ml(mli, rho)
    top = ml_top(ml)
    topf = filter_ml_top(top)
    c.data['MLI'] = mli
    c.data['ML'] = ml
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI', 'ML'], cmap='viridis')
    topf.plot(marker='_', linestyle='', ax=axarr[-3], color='red', label='ML top')
    #
    i = 5
    mlcol = ml.iloc[:, i]
    mlicol = mli.iloc[:, i]
    peaksi = mli.apply(peak_series)
    peaks = peaksi.apply(lambda i: list(mli.iloc[i].index))
    plot_peaks(peaks, ax=axarr[3])

