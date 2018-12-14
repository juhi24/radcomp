# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from sklearn import preprocessing


SCALING_LIMITS = {'ZH': (-10, 30), 'zh': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3),
                  'KDP': (0, 0.5), 'kdp': (0, 0.15)}


def scale(data, param='zh'):
    """radar data scaling"""
    scaled = data.copy()
    scaled -= SCALING_LIMITS[param][0]
    scaled *= 1.0/SCALING_LIMITS[param][1]
    return scaled


def scale_inv(data, param='zh'):
    """inverse radar data scaling"""
    scaled = data.copy()
    scaled *= SCALING_LIMITS[param][1]
    scaled += SCALING_LIMITS[param][0]
    return scaled


class RadarDataScaler(preprocessing.FunctionTransformer):
    """FunctionTransformer wrapper"""

    def __init__(self, param='zh', **kws):
        self.param = param
        fun_kws = dict(param=param)
        inv_kws = fun_kws
        super().__init__(func=scale, inverse_func=scale_inv,
                         kw_args=fun_kws, inv_kw_args=inv_kws, **kws)