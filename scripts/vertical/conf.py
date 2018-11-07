# coding: utf-8
"""Settings used in current studies."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path
from radcomp.vertical import multicase, RESULTS_DIR


#SCHEME_ID_SNOW = '14-16_t08_zdr05_19eig19clus_pca'
SCHEME_ID_SNOW = 'snow_t08_kdp13_22eig23clus_pca'
SCHEME_ID_MELT = 'mlt2_kdp08_18eig15clus_pca'

CASES_SNOW = 'snow'
CASES_MELT = 'melting'

PARAMS = ['zh', 'zdr', 'kdp']

VPC_PARAMS_SNOW = dict(basename='snow',
                       params=PARAMS,
                       hlimits=(190, 10e3),
                       n_eigens=0.8,
                       n_clusters=23,
                       reduced=True,
                       use_temperature=True,
                       t_weight_factor=0.8,
                       radar_weight_factors=dict(kdp=1.3))

VPC_PARAMS_RAIN = dict(basename='mlt2',
                       params=PARAMS,
                       hlimits=(290, 10e3),
                       n_eigens=0.8,
                       n_clusters=17,
                       reduced=True,
                       use_temperature=False,
                       radar_weight_factors=dict())

P1_FIG_DIR = path.join(RESULTS_DIR, 'paper1')


def init_cases(cases_id=None, season=''):
    """initialize cases data"""
    if cases_id is None:
        if season == 'snow':
            cases_id = CASES_SNOW
        elif season == 'rain':
            cases_id = CASES_MELT
    cases = multicase.read_cases(cases_id)
    if cases.ml.astype(bool).all():
        cases = cases[cases.ml_ok.astype(bool)]
    return cases

