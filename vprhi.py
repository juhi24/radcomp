#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from os import path

plt.ion()
plt.close('all')
sns.reset_orig()

datadir = '/home/jussitii/DATA/ToJussi'
datapath = path.join(datadir, '20140131_IKA_VP_from_RHI.mat')

def vprhimat2pn(datapath):
    data = scipy.io.loadmat(datapath)['VP_RHI']
    fields = list(data.dtype.fields)
    fields.remove('ObsTime')
    fields.remove('height')
    str2dt = lambda tstr: pd.datetime.strptime(tstr,'%Y-%m-%dT%H:%M:%S')
    t = map(str2dt, data['ObsTime'][0][0])
    h = data['height'][0][0][0]
    data_dict = {}
    for field in fields:
        data_dict[field] = data[field][0][0].T
    return pd.Panel(data_dict, major_axis=h, minor_axis=t)

pn = vprhimat2pn(datapath)

plt.pcolormesh(np.ma.masked_invalid(pn.ZH.values), cmap='gist_ncar', vmin=-2, vmax=30)
#plt.yticks(np.arange(0.5, len(pn.ZH.index), 1), pn.ZH.index)
#plt.xticks(np.arange(0.5, len(pn.ZH.columns), 1), pn.ZH.columns)


