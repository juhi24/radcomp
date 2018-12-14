# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from functools import partial

from sklearn import preprocessing


SCALING_LIMITS = {'ZH': (-10, 30), 'zh': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3),
                  'KDP': (0, 0.5), 'kdp': (0, 0.15)}


def scale(data, param='zh'):
    scaled = data.copy()
    scaled -= SCALING_LIMITS[param][0]
    scaled *= 1.0/SCALING_LIMITS[param][1]
    return scaled


def scale_inverse(data, param='zh'):
    scaled = data.copy()
    scaled *= SCALING_LIMITS[param][1]
    scaled += SCALING_LIMITS[param][0]
    return scaled


class RadarDataScaler(preprocessing.FunctionTransformer):
    def __init__(self, param, **kws):
        self.param = param
        func = partial(scale, param=param)
        inverse = partial(scale_inverse, param=param)
        super().__init__(func=func, inverse_func=inverse, **kws)