# coding: utf-8
"""Settings used in current studies."""

from os import path
from radcomp.vertical import classification, multicase, RESULTS_DIR

from j24 import ensure_join


CASES_SNOW = 'snow'
CASES_RAIN = 'rain'

PARAMS = ['zh', 'zdr', 'kdp']

VPC_PARAMS_SNOW = dict(basename='snow',
                       has_ml=False,
                       params=PARAMS,
                       hlimits=(190, 10e3),
                       n_eigens=30,
                       n_clusters=16,
                       reduced=True,
                       extra_weight=0.75)

VPC_PARAMS_RAIN = dict(basename='rain',
                       has_ml=True,
                       params=PARAMS,
                       hlimits=(290, 10e3),
                       n_eigens=30,
                       n_clusters=10,
                       reduced=True)

SEED = 0

SCHEME_ID_SNOW = classification.scheme_name(**VPC_PARAMS_SNOW)
SCHEME_ID_RAIN = classification.scheme_name(**VPC_PARAMS_RAIN)

P1_FIG_DIR = ensure_join(RESULTS_DIR, 'paper1')
POSTER_FIG_DIR = ensure_join(RESULTS_DIR, 'poster')


def init_cases(cases_id=None, season=''):
    """initialize cases data"""
    if cases_id is None:
        if season == 'snow':
            cases_id = CASES_SNOW
        elif season == 'rain':
            cases_id = CASES_RAIN
    cases = multicase.read_cases(cases_id)
    if cases.ml.astype(bool).all():
        cases = cases[cases['ml_ok'].fillna(0).astype(bool)]
    return cases

