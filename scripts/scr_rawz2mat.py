#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import pyart
import os
import scipy.io
import numpy as np

basepath = '/home/jussitii/DATA/radar'
resultspath = basepath
testpath = os.path.join(basepath, 'test')

fn1 = '20080627121802_KUM_ppi_LEP_B.raw'
fn2 = '20080902093219_KUM_ppi_POL_B.raw'

for i, fn in enumerate((fn1, fn2)):
    data = {}
    fnbase = os.path.splitext(fn)[0]
    fp = os.path.join(basepath, fn)
    radar = pyart.io.read(fp)
    mat_fp = os.path.join(resultspath, fnbase + '.mat')
    data['dbz'] = radar.get_field(0, 'reflectivity').data
    data['elev'] = radar.get_elevation(0)
    data['azim'] = radar.get_azimuth(0)
    data['range'] = radar.range['data']
    scipy.io.savemat(mat_fp, data)
    