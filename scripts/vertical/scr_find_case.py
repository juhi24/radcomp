# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baecc.instruments.pluvio as pl
from os import path
from glob import glob
from datetime import datetime, timedelta
from radcomp import CACHE_TMP_DIR
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import home

plt.ion()
plt.close('all')
np.random.seed(0)

case_set = 'everything'
n_eigens = 25
n_clusters = 20
reduced = True

DATADIR = path.join(home(), 'DATA')

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)

def dateparser(dstr):
    return datetime.strptime(dstr, '%y%m%d').date()

def time_start():
    vpdir = path.join(DATADIR, 'vprhi')
    datefile = path.join(vpdir, 'date.list')
    dates = pd.read_csv(datefile, dtype=str, header=None).iloc[:, 0]
    date_start = dates.apply(datetime.strptime, args=['%Y%m%d'])
    date_start.index = date_start.apply(lambda t: t.date())
    date_start.name = 'start'
    # January 2014 data is bad
    date_start = date_start.loc[datetime(2014, 2, 1).date():].copy()
    return date_start

def dates():
    date_start = time_start()
    date_end = date_start + timedelta(days=1) - timedelta(minutes=1)
    date_end.name = 'end'
    date_end.index = date_start.index
    dt2str = lambda t: t.strftime('%Y-%m-%d %H:%M')
    date = pd.concat([date_start.apply(dt2str), date_end.apply(dt2str)], axis=1)
    dates_t = pd.read_hdf(path.join(home(), 'DATA', 't_fmi_14-17.h5'), 'data').index
    sdates_t = pd.Series(dates_t, index=dates_t)
    uniqdates_t = sdates_t.apply(lambda t: t.date()).unique()
    return date.loc[uniqdates_t].dropna()

def everycase():
    date = dates()
    dtpath = path.join(home(), '.radcomp', 'cases', 'everything.csv')
    date.to_csv(dtpath, mode='w', index=False, header=True)

def pluvglobs(dates, pattern):
    pluvglobpatterns = dates.apply(lambda t: t.strftime(pattern))
    return pluvglobpatterns.apply(glob)

cases.index=list(map(dateparser, cases.index.values))
d = time_start()
p2patt = path.join(DATADIR, 'Pluvio200', 'pluvio200_0?_%Y%m%d??.txt')
p4patt = path.join(DATADIR, 'Pluvio400', 'pluvio400_0?_%Y%m%d??.txt')
p2globs = pluvglobs(d, p2patt)
p4globs = pluvglobs(d, p4patt)
