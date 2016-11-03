#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import pyart
import os
import matplotlib.pyplot as plt

plt.ion()

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
testpath = os.path.join(basepath, 'test')

def rawpath(sitename):
    return os.path.join(basepath, sitename + 'data/MonthlyArchivedData/2016-09/')

def rawfilepath(sitename, filename):
    return os.path.join(rawpath(sitename), filename)