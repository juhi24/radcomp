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


def savgol_series(data, *args, **kws):
    """savgol filter for Series"""
    result_arr = signal.savgol_filter(data.values.flatten(), *args, **kws)
    return pd.Series(index=data.index, data=result_arr)


def consecutive_grouper(s):
    return (s != s.shift()).cumsum()


def first_consecutive(s):
    """only first consecutive group of trues is kept"""
    grouper = consecutive_grouper(s)
    g = s.groupby(grouper)
    true_groups = g.mean()[g.mean()]
    if true_groups.empty:
        return grouper==-1
    return grouper==[true_groups.index[0]]


def expand_ml(ml, rho, rholim=0.975):
    """expand detected ml using rhohv"""
    grouper = consecutive_grouper(rho<rholim)
    selected = grouper[ml]
    if selected.empty:
        return ml
    iml = selected.iloc[0]
    ml_new = grouper==iml
    rho_above_ml = rho[grouper==iml+1].iloc[0]
    if (rho_above_ml < rholim) or np.isnan(rho_above_ml):
        return mlcol==np.inf # all False
    return ml_new


def ml_top(ml, maxh=4500, no_ml_val=np.nan):
    """extract ml top height from detected ml"""
    top = ml[::-1].idxmax()
    if no_ml_val is not None:
        top[top>maxh] = no_ml_val
    return top


def filter_ml_top(top, size=3):
    """apply median filter to ml top height"""
    topf = filtering.median_filter_df(top.fillna(0), size=size)
    topf[np.isnan(top)] = np.nan
    return topf


basename = 'melting-test'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 20
n_clusters = 20
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
    c = cases.case.iloc[2]
    c.class_scheme = scheme
    c.train()
    i = 27
    scaled_data = case.scale_data(c.data)
    zdr = scaled_data.zdr
    zh = scaled_data.ZH
    rho = c.data.RHO
    f, ax = plt.subplots()
    rhocol = rho.iloc[:, i]
    zdr.iloc[:, i].plot(ax=ax, label='ZDR')
    rhocol.plot(ax=ax, label='RHO')
    zh.iloc[:, i].plot(ax=ax, label='ZH')
    ax.set_title(zh.columns[i])
    indicator = (1-rho)*(zdr+1)*zh
    mli = indicator*100
    mli = mli.apply(savgol_series, args=(5, 2))
    c.data['MLI'] = mli
    indicatorcol = mli.iloc[:, i]
    c.data.MLI.iloc[:, i].plot(ax=ax, label='MLT')
    ax.legend()
    dz = pd.DataFrame(index=indicatorcol.index, data=ndimage.sobel(indicatorcol))
    a = filtering.median_filter_df(indicatorcol, size=4)
    #(a*10).plot(ax=ax)
    pp = peakutils.indexes(indicatorcol, min_dist=10)
    cwtp = signal.find_peaks_cwt(indicatorcol, range(1,10))
    plt.figure()
    pplot(indicatorcol.index.values, indicatorcol.values, pp)
    ml = c.data.MLI>2
    ml[rho>0.975] = False
    ml = ml.apply(first_consecutive)
    ml.iloc[:, i].astype(float).plot(ax=ax)
    # expand
    for t, mlc in ml.iteritems():
        ml[t] = expand_ml(mlc, rho[t])
    c.data['ML'] = ml
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI', 'ML'], cmap='viridis')
    mlcol = ml.iloc[:, i]
    mlicol = mli.iloc[:, i]
    top = ml_top(ml)
    topf = filter_ml_top(top)
    topf.plot(marker='_', linestyle='', ax=axarr[-3], color='red')

