# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd


def var_comb(cases, case_func):
    # TODO: duplicate in scr_class_scatter
    var_list = []
    for cname in cases.index:
        #data_g = g.loc[name]
        var_list.append(case_func(cases.case[cname]))
    return pd.concat(var_list)


def fr_comb(cases):
    func = lambda case: case.fr()
    return var_comb(cases, func)


def lwe_comb(cases):
    func = lambda case: case.lwe()
    lwe = var_comb(cases, func)
    lwe[lwe==np.inf] = 0
    return lwe


def t_comb(cases):
    func = lambda case: case.t_surface()
    return var_comb(cases, func)


def lwp_comb(cases):
    func = lambda case: case.lwp()
    return var_comb(cases, func)


def azs_comb(cases):
    func = lambda case: case.azs()
    return var_comb(cases, func)


def classes_comb(cases, name):
    def func(case):
        case.load_classification(name)
        classes = case.classes
        classes.index = classes.index.round('1min')
        classes.name = 'class'
        return classes
    return var_comb(cases, func)
