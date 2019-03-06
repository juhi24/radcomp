# coding: utf-8
"""Rearrange cases based on radar echos."""

import datetime

import pandas as pd
import numpy as np

from radcomp.vertical import multicase


def _icombine(comb, cases):
    ccases = cases.iloc[comb]
    c = multicase.MultiCase.by_combining(ccases)
    # fill gaps with nans
    c.data = c.data.resample('15min', axis=2, base=c.base_minute()).mean()
    return c


def _icomb_bool(comb, column, cases):
    conv = cases[column].iloc[comb]
    if conv.isnull().any():
        return np.nan
    if conv.any() != conv.all():
        return np.nan
        raise ValueError('Conflict in boolean flags.')
    return conv.astype(bool).any().astype(float)


def grouper_orig(cc):
    """grouper based on gaps in timestamps"""
    diffs = cc.timestamps().diff()
    grpr_orig = (diffs>cc.timedelta).cumsum()
    grpr_orig.name = 'g_orig'
    return grpr_orig


def echo_gaps(cc):
    """Series of timedeltas since previous echo"""
    ts = cc.timestamps()
    no_echo = cc.classes()==0
    i_no_echo = no_echo[no_echo].index
    ts.drop(i_no_echo, inplace=True)
    gaps = ts.diff()
    gaps.name = 'echo_gap'
    return gaps


def combine_cases_t_thresh(cases, gap=datetime.timedelta(hours=12)):
    """Combine cases with echo gaps less than threshold."""
    cc = multicase.MultiCase.by_combining(cases)
    grpr_orig = grouper_orig(cc)
    egaps = echo_gaps(cc)
    grouper_new = (egaps>gap).cumsum()+10000
    groupers = pd.concat([grouper_new, grpr_orig], axis=1)
    g_drop=groupers.dropna()
    g = g_drop.g_orig.groupby(g_drop.echo_gap)
    combinations = g.unique()
    case = combinations.apply(_icombine, args=[cases])
    case.index = case.apply(lambda x: x.name())
    combinations.index = case.index
    case.name = 'case'
    case.index.name = 'id'
    start = combinations.apply(lambda comb: cases.start.iloc[comb[0]])
    start.name = 'start'
    end = combinations.apply(lambda comb: cases.end.iloc[comb[-1]])
    end.name = 'end'
    ml = combinations.apply(_icomb_bool, args=['ml', cases])
    ml.name = 'ml'
    #comment = combinations.apply(lambda comb: '; '.join(cases.comment.iloc[comb].values))
    cases_new = pd.concat((start, end, ml, case), axis=1)
    if 'ml_ok' in cases.columns:
        cases_new['ml_ok'] = combinations.apply(_icomb_bool, args=['ml_ok', cases])
    if 'convective' in cases.columns:
        cases_new['convective'] = combinations.apply(_icomb_bool, args=['convective', cases])
    return cases_new, cc


if __name__ == '__main__':
    cases_s.case.loc['160110T21-14T06'].plot()