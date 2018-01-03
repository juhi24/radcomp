# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import case

if __name__ == '__main__':
    plt.close('all')
    cases = case.read_cases('melting')
    c = cases.case.iloc[0]
    c.plot(params=['ZH', 'zdr', 'kdp', 'RHO'], cmap='viridis')

