# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from sklearn import preprocessing


SCALING_LIMITS_SNOW = {'zh': (-10, 35), 'zdr': (0, 4), 'kdp': (0, 0.1)}
SCALING_LIMITS_RAIN = {'zh': (-10, 38), 'zdr': (0, 3), 'kdp': (0, 0.22)}


def scale(data, param='zh', has_ml=False, inverse=False):
    """radar data scaling"""
    scaled = data.copy()
    limits = SCALING_LIMITS_RAIN if has_ml else SCALING_LIMITS_SNOW
    if inverse:
        scaled *= limits[param][1]
        scaled += limits[param][0]
    else:
        scaled -= limits[param][0]
        scaled *= 1.0/limits[param][1]
    return scaled


class RadarDataScaler(preprocessing.FunctionTransformer):
    """FunctionTransformer wrapper"""

    def __init__(self, param='zh', has_ml=False, **kws):
        self.param = param
        self.has_ml = has_ml
        fun_kws = dict(param=param, has_ml=has_ml, inverse=False)
        inv_kws = dict(param=param, has_ml=has_ml, inverse=True)
        super().__init__(func=scale, inverse_func=scale,
                         kw_args=fun_kws, inv_kw_args=inv_kws, **kws)