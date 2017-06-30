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
from scr_find_case import plot_case, STORE_FILE, NAME, store

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
    #row = data.loc[datetime(2015, 4, 1).date()]
    row = data.loc[datetime(2014, 12, 24).date()]
    c = row.case
    fig, axarr = plot_case(c, row.pluvio400, cmap='viridis')
    axi = axarr[-2]
    axi.set_ylim(bottom=0, top=4)
    p4 = row.pluvio400
    i = p4.intensity()
    am = p4.amount()
    plotting.plot_data(i, ax=axi)
    iw = c.time_weighted_mean(i, offset_half_delta=True)
    ii = i.between_time('15:00', '17:00')
    iiw = iw.between_time('15:00', '17:00')
    plt.figure()
    iw.plot(drawstyle='steps')
    i.plot(drawstyle='steps')




