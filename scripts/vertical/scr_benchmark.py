# coding: utf-8
"""benchmarking vpc"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import benchmark


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    bm = benchmark.ProcBenchmark.from_csv()
    bm.fit(c.class_scheme)
    ax = bm.query_all(bm.query_count).plot.bar(stacked=True)
    ax.set_ylabel('number of profiles')
    ax.set_xlabel('class')
    ax.set_title('snow classification vs. manual analysis')