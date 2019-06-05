# coding: utf-8
"""Analyze profile class co-occurrence."""

import pandas as pd
import matplotlib.pyplot as plt

from radcomp.vertical import multicase


def cl_coocc(cases, cc, index_str=True, logical_or=False):
    """profile class co-occurrence"""
    ts = multicase.ts_case_ids(cases)
    cl = cc.classes()
    dat = pd.concat([ts, cl], axis=1).dropna()
    datu = dat.drop_duplicates()
    datu['class'] = datu['class'].astype(int)
    bmat = datu.groupby(['case', 'class']).size().unstack(fill_value=0)
    if logical_or:
        bmat = (-bmat.astype(bool)).astype(int)
    coocc = bmat.T.dot(bmat)
    if logical_or:
        coocc = cases.shape[0]-coocc
    if index_str:
        clarr = cc.vpc.class_labels()
        coocc.index = clarr
        coocc.columns = clarr
    return coocc


def imshow_coocc(coocc, percent=True, ax=None):
    """visualize profile class co-occurrence matrix"""
    ax = ax or plt.gca()
    size = coocc.shape[0]
    annot = (coocc*100).round().astype(int).values if percent else coocc.values
    ax.imshow(coocc.T)
    for x in range(size):
        for y in range(size):
            val = annot.T[x][y]
            color = 'white' if percent and (val < 50) else 'black'
            ax.annotate('{}'.format(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', color=color)
    ax.set_xticks(range(size))
    ax.set_xticklabels(coocc.index)
    ax.set_yticks(range(size))
    ax.set_yticklabels(coocc.index)
    return ax



if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    cases = cases_s
    cc = cc_s
    coocc = cl_coocc(cases, cc, logical_or=False)
    # co-occurrence fraction of all profiles
    coocc_frac = coocc/cases.shape[0]
    coocc_rel = coocc/coocc.values.diagonal()
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axrel = axarr[1]
    axfrac = axarr[0]
    imshow_coocc(coocc_frac, percent=True, ax=axfrac)
    imshow_coocc(coocc_rel, percent=True, ax=axrel)
    axrel.set_ylabel('B')
    axrel.set_xlabel('A')
    axrel.set_title('Frequency of A given presence of B')