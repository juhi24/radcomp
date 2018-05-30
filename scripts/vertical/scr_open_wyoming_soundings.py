# coding: utf-8
"""open soundings from wyoming website"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import webbrowser
import time
import pandas as pd
from datetime import timedelta
from radcomp.vertical import multicase, classification, sounding

case_set = 'melting'
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
name = 'mlt_18eig17clus_pca'

if __name__ == '__main__':
    search_class = 11
    c = multicase.MultiCase.from_caselist(case_set, filter_flag='ml_ok')
    c.load_classification(name)
    matchi = c.classes[c.classes==search_class].index
    matcha = pd.Series(data=matchi, index=matchi)
    match = pd.concat([matcha.iloc[0:1], matcha[matcha.diff()>timedelta(hours=2)]])
    for d, cla in match.iteritems():
        t = sounding.round_hours(d, hres=12)
        url = sounding.sounding_url(t, dtype='text')
        webbrowser.open(url)
        time.sleep(1)


