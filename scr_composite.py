#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""

import scipy.io
import glob
import os
import numpy as np
import pandas as pd
#import radx

basepath = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
resultspath = '/home/jussitii/results/radcomp'
intrp_path = os.path.join(basepath, 'interpolated')

def composite_max(maps):
    return np.maximum.reduce(maps)

times = ts=pd.date_range(pd.datetime(2016, 9, 3, 10), pd.datetime(2016, 9, 4, 6, 59), freq='Min').tolist()
mats = glob.glob(os.path.join(intrp_path, '???', 'R', 'mat', '20160903', 'intrp_20160903_2105??.mat'))
rs = [scipy.io.loadmat(fpath)['R'] for fpath in mats]
comp = composite_max(rs)