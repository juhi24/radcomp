# coding: utf-8
"""Plot a VP case before vs. after filtering."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, plotting, RESULTS_DIR

case_set = '14-16by_hand'
n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)

if __name__ == '__main__':
    plt.ion()
    #plt.close('all')
    cases = case.read_cases(case_set)
    c = cases.case.loc['140221-22']
    raw = case.data_range(c.t_start(), c.t_end())
    pair = pd.Panel.from_dict(dict(KDP=raw.KDP, kdp=c.data.kdp))
    fig, axarr = plotting.plotpn(pair, cmap='viridis')
    axarr[0].set_title('$K_{dp}$ before and after filtering')
    for ax in axarr:
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
    fig.savefig(path.join(RESULTS_DIR, 'filtering_example.png'))


