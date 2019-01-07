# coding: utf-8
"""benchmarking vpc"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, benchmark

import conf


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    bm = benchmark.ProcBenchmark.from_csv(fltr_q='~ml')
    #c = multicase.MultiCase.from_caselist('snow')
    #c.load_classification(conf.SCHEME_ID_SNOW)
    bm.fit(c.class_scheme)
    ax = bm.query_all(bm.query_count).plot.bar(stacked=True)
    ax.grid(axis='y')
    ax.set_ylabel('number of profiles')
    ax.set_xlabel('class')
    ax.set_title('snow classification vs. manual analysis')