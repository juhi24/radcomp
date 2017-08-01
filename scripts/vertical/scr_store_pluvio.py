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

plt.ioff()
plt.close('all')
np.random.seed(0)

CASE_SET = 'everything'
n_eigens = 25
n_clusters = 20
reduced = True

NAME = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
DATADIR = path.join(home(), 'DATA')
STORE_FILE = path.join(CACHE_TMP_DIR, CASE_SET + '.hdf')
fig_dir = ensure_join(RESULTS_DIR, 'class_r_t', NAME, CASE_SET)


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
    dates_t = pd.read_hdf(path.join(DATADIR, 't_fmi_14-17.h5'), 'data').index
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
    cases = case.read_cases(CASE_SET)
    cases.index=list(map(dateparser, cases.index.values))
    df2 = pluvs(pluvtype='200')
    df4 = pluvs(pluvtype='400')
    data = pd.concat([cases, df2, df4], join='inner', axis=1)
    data.to_hdf(store_file, 'data')


def plot_case(c, *pluvs, **kws):
    try:
        c.load_classification(NAME)
    except ValueError:
        warn('ValueError while trying classification. Skipping case.')
        return None, None
    half_dt = c.mean_delta()/2
    fig, axarr = c.plot(n_extra_ax=1, **kws)
    axi = axarr[-2]
    for pluv in pluvs:
        i = pluv.intensity()
        iw = c.time_weighted_mean(i).shift(freq=half_dt)
        plotting.plot_data(iw, ax=axi, label=pluv.name)
    axi.yaxis.grid(True)
    axi.legend()
    axi.set_ylim(bottom=0, top=4)
    axi.set_ylabel(plotting.LABELS['intensity'])
    c.set_xlim(axi)
    for ax in axarr:
        ax.xaxis.grid(True)
    return fig, axarr


if __name__ == '__main__':
    data = pd.read_hdf(STORE_FILE)
    for t, row in data.iterrows():
        c = row.case
        print(c.name())
        fig, axarr = plot_case(c, row.pluvio200, row.pluvio400, cmap='viridis')
        if fig is None:
            continue
        fig.savefig(path.join(fig_dir, c.name()+'.png'))
        plt.close(fig)




