# coding: utf-8
"""Filter ZDR using rhohv."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import case, filtering

if __name__ == '__main__':
    plt.close('all')
    cases = case.read_cases('zdr_clutter')
    for i, c in cases.case.iteritems():
        c.plot(params=['ZH', 'ZDR', 'zdr', 'RHO'], cmap='viridis')

