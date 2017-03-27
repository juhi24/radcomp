# coding: utf-8
import pickle
import numpy as np
import pandas as pd
from os import path
from baecc import prepare
from radcomp import CACHE_DIR

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

def time_weighted_mean(data, rule='15min', param='intensity', offset=0, **kws):
    resampler = data.resample(rule=rule, loffset=pd.Timedelta(offset), **kws)
    return resampler.agg(weighted_avg, param=param)[param]

def weighted_avg(df, param='intensity'):
    out = df[param].dropna()
    w = weights(df.tdelta.dropna())
    if w.size < 1:
        return np.nan
    return np.average(out, weights=w)

def weights(tdelta):
    return tdelta.fillna(0).astype(int)
