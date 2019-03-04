# coding: utf-8
"""Rearrange cases based on radar echos."""

import datetime

import pandas as pd

from radcomp.vertical import multicase


def _icombine(comb, cases):
    ccases = cases.iloc[comb]
    return multicase.MultiCase.by_combining(ccases)


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
    cases_new = combinations.apply(_icombine, args=[cases])
    cases_new.index=cases_new.apply(lambda x: x.name())
    cases_new.name = 'case'
    return cases_new