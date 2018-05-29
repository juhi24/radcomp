# coding: utf-8
"""A script to analyze a chosen class."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import multicase


def load_ml_cases(cases_name='melting', scheme_name='mlt_18eig17clus_pca'):
    """load cases and classification"""
    cases = multicase.read_cases(cases_name)
    cases = cases[cases['ml_ok'].astype(bool)] # only verified ones
    cases.case.apply(lambda c: c.load_classification(scheme_name))
    return cases


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    class_n = 7
    cases = load_ml_cases()

