#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import pyart
import os
import scipy.io

basepath = '/home/jussitii/DATA/radar'
resultspath = basepath
testpath = os.path.join(basepath, 'test')

fn1 = '20080627121802_KUM_ppi_LEP_B.raw'
fn2 = '20080902093219_KUM_ppi_POL_B.raw'

for i, fn in enumerate((fn1, fn2)):
    fnbase = os.path.basename(fn)
    fp = os.path.join(basepath, fn)
    radar = pyart.io.read(fp)
    relf_fp = os.path.join(resultspath, fnbase + '_refl.mat')
    range_fp = os.path.join(resultspath, fnbase + '_range.mat')
    azim_fp = os.path.join(resultspath, fnbase + '_azim.mat')
    elev_fp = os.path.join(resultspath, fnbase + '_elev.mat')
    scipy.io.savemat(relf_fp, radar.fields['reflectivity'])
    scipy.io.savemat(azim_fp, radar.azimuth)
    scipy.io.savemat(elev_fp, radar.elevation)
    scipy.io.savemat(range_fp, radar.range)
    