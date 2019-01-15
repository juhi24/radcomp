# coding: utf-8
"""benchmarking vpc"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
import numpy as np

from radcomp.vertical import classification, benchmark, plotting, multicase

import conf


def check_man_cl(c, bm, fields=['zh', 'zdr', 'kdp'], cmap='viridis', **kws):
    for proc in {'hm', 'dgz'}:
        data = c.data.loc[:, :, bm.data_fitted[bm.data_fitted[proc]].index]
        data_p = data.copy()
        data_p.minor_axis = range(data.shape[2])
        plotting.plotpn(data_p, x_is_date=False, fields=fields, cmap=cmap, **kws)


def with_zdr(bm):
    stat = bm.query_all(bm.query_count, procs={'hm_kdp', 'dgz_kdp', 'dgz_zdr'})
    cval = 0.8
    cno = 0
    colors = [(cval,cno,cno), (cno,cval,cno), (cno,cno,cval),
              (cval,cval,cno), (cval,cno,cval), (cno,cval,cval),
              (cval, cval, cval)]
    fig, ax = plt.subplots()
    plotting.plot_bm_stats(stat, ax=ax, color=colors)
    return stat, ax


def just_kdp(bm):
    stat = bm.query_all(bm.query_count, procs={'hm_kdp', 'dgz_kdp'})
    fig, ax = plt.subplots()
    plotting.plot_bm_stats(stat.drop('non-event', axis=1), ax=ax)
    return stat, ax


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    rain_season = False
    casesname = conf.CASES_MELT if rain_season else conf.CASES_SNOW
    flag = 'ml_ok' if rain_season else None
    #c = multicase.MultiCase.from_caselist(casesname, filter_flag=flag)
    #bm = benchmark.ManBenchmark.from_csv(fltr_q='ml')
    vpc_params = conf.VPC_PARAMS_RAIN if rain_season else conf.VPC_PARAMS_SNOW
    for n_clusters in np.arange(8, 18):
        vpc_params.update({'n_clusters': n_clusters})
        name = classification.scheme_name(**vpc_params)
        vpc = classification.VPC.load(name)
        c.classify(scheme=vpc)
        ref = benchmark.autoref(c.data, rain_season=rain_season)[c.silh_score>0]
        bm = benchmark.AutoBenchmark(ref)
        bm.fit(vpc)
        stat, ax = just_kdp(bm)
