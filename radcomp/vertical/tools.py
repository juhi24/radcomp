# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
"""
Vertical profile classification
@author: Jussi Tiira
"""
import numpy as np


def m2km(m, pos):
    '''formatting m in km'''
    return '{:.1f}'.format(m*1e-3)


def echo_top_h(z, zmin=-8):
    top = (z>zmin).loc[::-1].idxmax()
    top[top==z.index[-1]] = np.nan
    return top
