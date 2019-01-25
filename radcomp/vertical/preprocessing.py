# coding: utf-8

from sklearn import preprocessing


SCALING_LIMITS_SNOW = {'zh': (-10, 34), 'zdr': (0, 3.3), 'kdp': (0, 0.11)}
SCALING_LIMITS_RAIN = {'zh': (-10, 38), 'zdr': (0, 3.1), 'kdp': (0, 0.25),
                       'kdpg': (-0.005, 0.005)}


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