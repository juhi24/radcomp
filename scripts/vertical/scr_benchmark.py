# coding: utf-8
"""benchmarking vpc"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import classification, benchmark, plotting

import conf


def check_man_cl(c, bm, fields=['zh', 'zdr', 'kdp'], cmap='viridis', **kws):
    for proc in {'hm', 'dgz'}:
        data = c.data.loc[:, :, bm.data_fitted[bm.data_fitted[proc]].index]
        data_p = data.copy()
        data_p.minor_axis = range(data.shape[2])
        plotting.plotpn(data_p, x_is_date=False, fields=fields, cmap=cmap, **kws)


def data_by_bm(bm, c):
    return c.data.loc[:, :, bm.data_fitted.index]


def prefilter(bm, c):
    """radar data based benchmark filter"""
    data = data_by_bm(bm, c)
    cond = data['kdp'].max() > 0.15 # prefilter condition
    bm.data_fitted = bm.data_fitted[cond].copy()


if __name__ == '__main__':
    plt.ion()
    #plt.close('all')
    bm = benchmark.ProcBenchmark.from_csv(fltr_q='ml')
    vpc = classification.VPC.load(conf.SCHEME_ID_MELT)
    bm.fit(vpc)
    stat = bm.query_all(bm.query_count)
    ax = stat.plot.bar(stacked=True)
    ax.grid(axis='y')
    ax.set_ylabel('number of profiles')
    ax.set_xlabel('class')
    ax.set_title('snow classification vs. manual analysis')