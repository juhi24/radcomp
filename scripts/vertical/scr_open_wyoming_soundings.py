# coding: utf-8
"""open soundings from wyoming website"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import webbrowser
import time
import pandas as pd
from radcomp import sounding
from radcomp.vertical import multicase, classification

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
    mc = multicase.MultiCase.from_caselist(case_set, filter_flag='ml_ok')
    mc.load_classification(name)
    matchi = mc.classes[mc.classes==search_class].index
    matcha = pd.Series(data=matchi, index=matchi)
    match = matcha.apply(sounding.round_hours, hres=12).drop_duplicates()
    for i, t in match.iteritems():
        url = sounding.sounding_url(t, dtype='text')
        webbrowser.open(url)
        time.sleep(1)


