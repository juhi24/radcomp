#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Vertical profile classification
@author: Jussi Tiira
"""
import pandas as pd
from os import path
from radcomp import USER_DIR
from radcomp.vertical.case import Case

def case_id_fmt(t):
    return t.strftime('%b%-d').lower()

def read_case_times(name):
    filepath = path.join(USER_DIR, 'cases', name + '.csv')
    dts = pd.read_csv(filepath, parse_dates=['t_start', 't_end'])
    dts.index = dts['t_start'].apply(case_id_fmt)
    dts.index.name = 'id'
    return dts

def read_cases(name):
    dts = read_case_times(name)
    cases_list = [Case.from_dtrange(row[1]['t_start'], row[1]['t_end']) for row in dts.iterrows()]
    dts['case'] = cases_list
    return dts

def m2km(m, pos):
    '''formatting m in km'''
    return '{:.0f}'.format(m*1e-3)

