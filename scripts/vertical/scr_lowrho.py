# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import multicase

if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    params = ['ZH', 'ZDR', 'KDP', 'RHO']
    cases = multicase.read_cases('lowrho')
    for cid, c in cases.case.iteritems():
        fig, _ = c.plot(plot_fr=False, plot_t=False, plot_azs=False,
                        plot_snd=False, cmap='viridis', params=params)
