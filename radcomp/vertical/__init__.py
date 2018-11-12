# coding: utf-8
"""tools for clustering vertical profiles of polarimetric radar variables"""
from os import path

home = path.expanduser('~')
NAN_REPLACEMENT = {'ZH': -10, 'ZDR': 0, 'KDP': 0, 'RHO': 0, 'DP': 0,
                   'KDP_ORIG': 0, 'PHIDP': 0, 'MLI': 0}
RESULTS_DIR = path.join(home, 'results', 'radcomp', 'vertical')

from radcomp.vertical.tools import m2km, echo_top_h
