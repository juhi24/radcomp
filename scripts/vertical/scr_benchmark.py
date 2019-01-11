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


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    #c = multicase.MultiCase.from_caselist('rain', filter_flag='ml_ok')
    #bm = benchmark.ManBenchmark.from_csv(fltr_q='ml')
    bm = benchmark.AutoBenchmark(benchmark.autoref(c.data))
    vpc_params = conf.VPC_PARAMS_RAIN
    for n_clusters in np.arange(10, 20):
        vpc_params.update({'n_clusters': n_clusters})
        name = classification.scheme_name(**vpc_params)
        vpc = classification.VPC.load(name)
        bm.fit(vpc)
        benchmark.prefilter(bm, c)
        stat = bm.query_all(bm.query_count,
                            procs={'hm_kdp', 'dgz_kdp', 'dgz_zdr'})
        ax = stat.plot.bar(stacked=True)
        ax.grid(axis='y')
        ax.set_ylabel('number of profiles')
        ax.set_xlabel('class')
        ax.set_title('unsupervised classification vs. manual analysis')