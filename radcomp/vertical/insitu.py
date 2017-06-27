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

def nextval(df, time):
    """Return value of df, whose index is closest higher to time."""
    try:
        return df[df.index>time][0]
    except IndexError:
        return np.nan

def _tw_mean(data, rule='15min', offset=None, **kws):
    if offset is None:
        offset = rule
    name = data.name
    t = pd.Series(data.index, index=data.index)
    tdelta = t.diff().dropna()
    tdelta.name = 'tdelta'
    dat = pd.concat([data, tdelta], axis=1, join='inner')
    resampler = dat.resample(rule=rule, loffset=pd.Timedelta(offset), **kws)
    return resampler.agg(weighted_avg, name=name)[name]

def time_weighted_mean(data, rule='15min', **kws):
    out = _tw_mean(data, rule=rule, **kws)
    t = pd.Series(data=out.index, index=out.index)
    prev = t.shift(freq=rule).iloc[:-1]
    prev.name = 'previous'
    df = pd.concat([out, prev], axis=1)
    easy = df.apply(lambda row: data.loc[row.previous:row.name].size<1, axis=1)
    out.loc[easy] = t.loc[easy].apply(lambda k: nextval(data, k))
    return out

def weighted_avg(data, name):
    out = data[name].dropna()
    w = weights(data.tdelta.dropna())
    if w.size < 1:
        return np.nan
    return np.average(out, weights=w)

def weights(tdelta):
    return tdelta.fillna(0).astype(int)
