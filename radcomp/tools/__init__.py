# coding: utf-8

from functools import partial
from datetime import date

import numpy as np
import pandas as pd


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)


def strftime_date_range(dt_start, dt_end, fmt):
    """strftime for a range of dates"""
    dt_range = pd.date_range(dt_start.date(), dt_end.date())
    dt2path_map = partial(date.strftime, format=fmt)
    return map(dt2path_map, dt_range)


