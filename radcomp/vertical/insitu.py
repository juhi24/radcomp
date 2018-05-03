# coding: utf-8
import pickle
import pandas as pd
from os import path
from baecc import prepare
from radcomp import CACHE_DIR
from j24 import home

TABLE_PKL = path.join(CACHE_DIR, 'insitu_table.pkl')
TABLE_FILTERED_PKL = path.join(CACHE_DIR, 'insitu_table_fltrd.pkl')
EVENTS_PKL = path.join(CACHE_DIR, 'insitu_events.pkl')

def store_insitu(casesname_baecc='tiira2017_baecc',
                      casesname_1415='tiira2017_1415', cases=None, **kws):
    e = prepare.events(casesname_baecc=casesname_baecc,
                       casesname_1415=casesname_1415)
    table_fltr = prep_table(e, cases=cases, **kws)
    table = prep_table(e, cases=cases, cond=None, **kws)
    table_fltr.to_pickle(TABLE_FILTERED_PKL)
    table.to_pickle(TABLE_PKL)
    if cases is not None:
        e.events.index = cases.index
    with open(EVENTS_PKL, 'wb') as f:
        pickle.dump(e, f)
    return table

def load_events():
    with open(EVENTS_PKL, 'rb') as f:
        return pickle.load(f)

def prep_table(e, cases=None, **kws):
    table = prepare.param_table(e=e, **kws)
    table.index = table.index.droplevel()
    if cases is not None:
        table.index.set_levels(cases.index, level=0, inplace=True)
    return table

def time_weighted_mean(data, rule='15min', resolution='1min', offset=None,
                       **kws):
    if offset is None:
        offset = rule
    upsampled = data.resample(rule=resolution).bfill()
    return upsampled.resample(rule=rule, loffset=pd.Timedelta(offset), **kws).mean()


def load_pluvio(start=None, end=None, kind='400'):
    """Load Pluvio data from hdf5 database."""
    import baecc.instruments.pluvio as pl
    name = 'pluvio{}'.format(str(kind))
    hdfpath = path.join(home(), 'DATA', 'pluvio14-16.h5')
    data = pd.read_hdf(hdfpath, key=name)[start:end]
    pluv = pl.Pluvio(data=data, name=name)
    return pluv

