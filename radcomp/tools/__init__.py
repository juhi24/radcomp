# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)
