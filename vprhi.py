#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import matplotlib as mpl
from os import path

plt.ion()
plt.close('all')

datadir = '/home/jussitii/DATA/ToJussi'
datapath0 = path.join(datadir, '20140131_IKA_VP_from_RHI.mat')
datapath1 = path.join(datadir, '20140201_IKA_VP_from_RHI.mat')

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

def plotpn(pn, fields=['ZH', 'ZDR', 'KDP']):
    vmin = {'ZH': -15, 'ZDR': -1, 'RHO': 0, 'KDP': 0}
    vmax = {'ZH': 30, 'ZDR': 4, 'RHO': 1, 'KDP': 0.26}
    label = {'ZH': 'dBZ', 'ZDR': 'dB', 'KDP': 'deg/km'}
    fig, axarr = plt.subplots(len(fields), sharex=True, sharey=True)
    def m2km(m, pos):
        return '{:.0f}'.format(m*1e-3)
    for i, field in enumerate(fields):
        ax = axarr[i]
        im = ax.pcolormesh(pn[field].columns, pn[field].index, 
                      np.ma.masked_invalid(pn[field].values), cmap='gist_ncar',
                      vmin=vmin[field], vmax=vmax[field], label=field)
        #fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
        ax.set_ylim(0,10000)
        ax.set_ylabel('Height, km')
        fig.colorbar(im, ax=ax, label=label[field])
    ax.set_xlabel('Time, UTC')
    axarr[0].set_title(str(pn[field].columns[0].date()))
    fig.tight_layout()
    return fig, axarr

pn0 = vprhimat2pn(datapath0)
pn1 = vprhimat2pn(datapath1)
pn = pd.concat([pn0, pn1],axis=2)
fig, axarr = plotpn(pn)


