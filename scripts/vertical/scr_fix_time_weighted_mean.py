# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baecc.instruments.pluvio as pl
from os import path
from glob import glob
from warnings import warn
from datetime import datetime, timedelta
from radcomp import CACHE_TMP_DIR
from radcomp.vertical import insitu, case, classification, plotting, RESULTS_DIR
from j24 import home, ensure_join
from scr_find_case import plot_case, STORE_FILE, NAME

plt.ion()
plt.close('all')
np.random.seed(0)

def nextval(df, time):
    try:
        return df[df.index>time].iloc[:1]
    except IndexError:
        return np.nan


def prevval(df, time):
    try:
        return ii[ii.index<time].iloc[-1:]
    except IndexError:
        return np.nan


def vals_around(df, time):
    post = nextval(df, time)
    pre = prevval(df, time)
    return pd.concat([pre, post])


if __name__ == '__main__':
    data = pd.read_hdf(STORE_FILE)
    row=data.loc[datetime(2015, 1, 7).date()]
    c = row.case
    fig, axarr = plot_case(c, row.pluvio200, row.pluvio400)
    axi = axarr[-2]
    axi.set_ylim(bottom=0, top=1.5)
    p4 = row.pluvio400
    i = p4.intensity()
    iw = c.time_weighted_mean(i)
    ii = i.between_time('11:00', '14:00')
    iiw = iw.between_time('11:00', '14:00')
    t = pd.Series(data=iiw.index, index=iiw.index)
    prev = t.shift(freq='15min').iloc[:-1]
    prev.name = 'previous'
    df = pd.concat([iiw, prev], axis=1)
    tw = iiw.index[4]
    to_avg = vals_around(ii, tw)
    r = df.iloc[4]
    t = ii.loc[r.previous:r.name].index[0]
    tt = vals_around(iiw, t).index
    w = abs(tt-t).values.astype(int)
    val = np.average(to_avg, weights=w)
    plt.figure()
    iiw.plot(drawstyle='steps')
    ii.plot(drawstyle='steps')







