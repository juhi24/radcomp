# coding: utf-8
"""melting layer detection sandbox"""

import os
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot
from radcomp.vertical import case
from j24 import home


def dbz_is_low(dbz):
    return dbz < 1


def rhohv_is_low(rhohv):
    return rhohv < 0.97


def drop_short_bool_sequence(df_bool, limit=5):
    """Drop sequences of true that are shorter than limit."""
    return df_bool.rolling(limit, center=True).sum() > (limit-0.5)


def find_dips(s, thres=0.04, **kws):
    """find lows

    optimized for finding melting layer signal in rhohv

    Returns:
        array: indices corresponding the dip bottoms"""
    y = (-s.fillna(0)).values
    return list(peakutils.peak.indexes(y, thres=thres, **kws))


def melting_px_candidate(pn):
    low_rhohv = rhohv_is_low(pn['RHO'])
    low_dbz = dbz_is_low(pn['ZH'])
    melting = low_rhohv & -low_dbz
    melting = drop_short_bool_sequence(melting)
    return melting


def dips_xy(dips, i):
    """x y pairs from dataframe"""
    x = np.full(len(dips[i]), dips.index[i])
    y = dips[i]
    return x, y


if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    cases = case.read_cases('melting-test')
    c = cases.case.iloc[1]
    c_snow = cases.case.iloc[0] # no melting
    rhohv = c.data['RHO']
    rhohv_snow = c_snow.data['RHO']
    rho_sample = rhohv.iloc[:,20]
    rho_snow_sample = rhohv_snow.iloc[:,20]
    c.data['MLT'] = melting_px_candidate(c.data)
    c_snow.data['MLT'] = melting_px_candidate(c_snow.data)
    fig, axarr = c.plot(params=['ZH', 'zdr', 'kdp', 'RHO', 'MLT'], cmap='viridis')
    #c_snow.plot(params=['ZH', 'zdr', 'kdp', 'RHO', 'MLT'], cmap='viridis')
    indices = find_dips(rho_sample)
    #plt.figure()
    #pplot(rho_sample.index, rho_sample.values, indices)
    dipsi = rhohv.apply(find_dips)
    dips = dipsi.apply(lambda i: list(rhohv.iloc[i].index))
    x = rhohv.columns[20]
    y = rhohv.iloc[dipsi.iloc[20], 20].index[0]
    for ts, vals in dips.iteritems():
        x = np.full(len(vals), ts)
        axarr[-2].scatter(x, vals, marker='+', zorder=1, color='red')

