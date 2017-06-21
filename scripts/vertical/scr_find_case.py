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
from radcomp.vertical import insitu, case, classification, plotting, RESULTS_DIR
from j24 import home

plt.ion()
plt.close('all')
np.random.seed(0)

case_set = 'everything'
n_eigens = 25
n_clusters = 20
reduced = True

DATADIR = path.join(home(), 'DATA')
STORE_FILE = path.join(CACHE_TMP_DIR, 'cases.hdf')


def dateparser(dstr):
    return datetime.strptime(dstr, '%y%m%d').date()


def time_start():
    vpdir = path.join(DATADIR, 'vprhi')
    datefile = path.join(vpdir, 'date.list')
    dates = pd.read_csv(datefile, dtype=str, header=None).iloc[:, 0]
    date_start = dates.apply(datetime.strptime, args=['%Y%m%d'])
    date_start.index = date_start.apply(lambda t: t.date())
    date_start.name = 'start'
    start = datetime(2014, 2, 1).date()
    end = datetime(2016, 5, 1).date()
    date_start = date_start.loc[start:end].copy()
    return date_start


def dates():
    date_start = time_start()
    date_end = date_start + timedelta(days=1) - timedelta(minutes=1)
    date_end.name = 'end'
    date_end.index = date_start.index
    date = pd.concat([date_start, date_end], axis=1)
    dates_t = pd.read_hdf(path.join(home(), 'DATA', 't_fmi_14-17.h5'), 'data').index
    sdates_t = pd.Series(dates_t, index=dates_t)
    uniqdates_t = sdates_t.apply(lambda t: t.date()).unique()
    return date.loc[uniqdates_t].dropna()


def dates_str():
    d = dates()
    dt2str = lambda t: t.strftime('%Y-%m-%d %H:%M')
    return d.apply(lambda row: row.apply(dt2str))


def everycase():
    date = dates_str()
    dtpath = path.join(home(), '.radcomp', 'cases', 'everything.csv')
    date.to_csv(dtpath, mode='w', index=False, header=True)


def pluvglobs(dates, pattern):
    pluvglobpatterns = dates.apply(lambda t: t.strftime(pattern))
    globs = pluvglobpatterns.apply(glob)
    return globs.loc[globs.apply(lambda l: len(l)>1)]


def pluvs(pluvtype='200'):
    pluvtype = str(pluvtype)
    d = dates()['start']
    fnamefmt = 'pluvio{}_0?_%Y%m%d??.txt'.format(pluvtype)
    patt = path.join(DATADIR, 'Pluvio{}'.format(pluvtype), fnamefmt)
    globs = pluvglobs(d, patt)
    df = globs.apply(pl.Pluvio)
    df.name = 'pluvio{}'.format(pluvtype)
    return df


def store(store_file=STORE_FILE):
    cases = case.read_cases(case_set)
    cases.index=list(map(dateparser, cases.index.values))
    df2 = pluvs(pluvtype='200')
    df4 = pluvs(pluvtype='400')
    data = pd.concat([cases, df2, df4], join='inner', axis=1)
    data.to_hdf(store_file, 'data')


if __name__ == '__main__':
    name = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                      n_clusters=n_clusters, reduced=reduced)
    data = pd.read_hdf(STORE_FILE)
    row = data.iloc[7] # 14
    c = row.case
    c.load_classification(name)
    i2 = row.pluvio200.intensity()
    i4 = row.pluvio400.intensity()
    half_dt=c.mean_delta()/2
    iw2 = c.time_weighted_mean(i2).shift(freq=-half_dt)
    iw4 = c.time_weighted_mean(i4).shift(freq=-half_dt)
    fig, axarr = c.plot(n_extra_ax=1)
    axi = axarr[-2]
    plotting.plot_data(iw2, ax=axi, label='pluvio200')
    plotting.plot_data(iw4, ax=axi, label='pluvio400')
    axi.yaxis.grid(True)
    axi.legend()
    for ax in axarr:
        ax.xaxis.grid(True)
        c.set_xlim(ax)







