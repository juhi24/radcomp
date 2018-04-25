# coding: utf-8
"""melting layer detection using scipy peak utilities"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from scipy import signal
import pandas as pd


def savgol_series(data, *args, **kws):
    """savgol filter for Series"""
    result_arr = signal.savgol_filter(data.values.flatten(), *args, **kws)
    return pd.Series(index=data.index, data=result_arr)


def indicator(zdr_scaled, zh_scaled, rho, savgol_args=(15, 3)):
    """Calculate ML indicator."""
    mli = (1-rho)*(zdr_scaled+1)*zh_scaled*100
    mli = mli.apply(savgol_series, args=savgol_args)
    return mli
