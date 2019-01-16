# coding: utf-8
"""retrieval of secondary variables derived from radar or model data"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np


def echotop(df):
    """highest index of each column with non-null data"""
    isnull = df.isnull()
    hghts = isnull.copy()
    for col in hghts.columns:
        hghts[col] = hghts.index
    hghts[isnull] = np.nan
    top = hghts.idxmax()
    top.name = 'echotop'
    return top