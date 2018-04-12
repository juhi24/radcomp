# coding: utf-8
"""implementing Brandon's ML detection method"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from radcomp.vertical import case, classification, filtering


basename = 'melting-test'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 20
n_clusters = 20
reduced = True
use_temperature = False
t_weight_factor = 0.8

scheme = classification.VPC(params=params, hlimits=hlimits, n_eigens=n_eigens,
                            n_clusters=n_clusters,
                            reduced=reduced, t_weight_factor=t_weight_factor,
                            basename=basename, use_temperature=use_temperature)

if __name__ == '__main__':
    plt.close('all')
    plt.ion()
    cases = case.read_cases('melting-test')
    c = cases.case.iloc[1]
    c.class_scheme = scheme
    c.train()
    fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO'], cmap='viridis')
    i = 29
    zdrcol = c.cl_data_scaled.zdr.T.iloc[:, i]
    rhocol = c.data.RHO.iloc[:, i]
    f, ax = plt.subplots()
    zdrcol.plot(ax=ax)
    rhocol.plot(ax=ax)
    indicator = (1-rhocol)*zdrcol
    (indicator*10).plot(ax=ax)
    dz = pd.DataFrame(index=indicator.index, data=ndimage.sobel(indicator))
    a=filtering.median_filter_df(indicator, size=4)
    (a*10).plot(ax=ax)
    # maybe rather filter dz?

