# coding: utf-8

import datetime

import pandas as pd


if __name__ == '__main__':
    diffs = cc.timestamps().diff()
    grouper_orig = (diffs>cc.timedelta).cumsum()
    grouper_orig.name = 'g_orig'
    orig_gaps = diffs[diffs>cc.timedelta]
    gap = datetime.timedelta(hours=12)
    ts = cc.timestamps()
    no_echo = cc.classes()==0
    i_no_echo = no_echo[no_echo].index
    ts.drop(i_no_echo, inplace=True)
    echo_gaps = ts.diff()
    echo_gaps.name = 'echo_gap'

    b=pd.concat([echo_gaps, grouper_orig], axis=1).dropna()
    c=b.shift()

    grouper_new = (echo_gaps>gap).cumsum()+1000
    groupers = pd.concat([grouper_new, grouper_orig], axis=1)
    g_drop=groupers.dropna()
    g=g_drop.g_orig.groupby(g_drop.echo_gap)
    combinations = g.unique()
