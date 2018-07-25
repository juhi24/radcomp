# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase
from scr_erad_cases import savedir

if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases('feb21')
    c = cases.case.iloc[0]
    fig, _ = c.plot(plot_fr=False, plot_t=False, plot_azs=False)
    fig.savefig(path.join(savedir, 'feb21.png'), bbox_inches='tight')
