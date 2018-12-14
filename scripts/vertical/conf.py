# coding: utf-8
"""Settings used in current studies."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path
from radcomp.vertical import multicase, RESULTS_DIR


#SCHEME_ID_SNOW = '14-16_t08_zdr05_19eig19clus_pca'
#SCHEME_ID_SNOW = 'snow_t08_kdp13_22eig23clus_pca'
SCHEME_ID_SNOW = 'snow_t08_kdp17_30eig12clus_pca'
SCHEME_ID_MELT = 'mlt2_kdp08_30eig10clus_pca'

CASES_SNOW = 'snow'
CASES_MELT = 'rain'

PARAMS = ['zh', 'zdr', 'kdp']

VPC_PARAMS_SNOW = dict(basename='snow',
                       has_ml=False,
                       params=PARAMS,
                       hlimits=(190, 10e3),
                       n_eigens=30,
                       n_clusters=12,
                       reduced=True,
                       extra_weight=0.8)

VPC_PARAMS_RAIN = dict(basename='mlt2',
                       has_ml=True,
                       params=PARAMS,
                       hlimits=(290, 10e3),
                       n_eigens=30,
                       n_clusters=10,
                       reduced=True)

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

