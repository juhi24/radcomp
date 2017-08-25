# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
from glob import glob
from os import path
from scipy.io import loadmat
from j24 import home, mldatenum2datetime

datadir = path.join(home(), 'DATA', 'BAECC_1308_AVL')
dataset = set(glob(path.join(datadir, 'Snow_*.mat')))
P200SET = set(glob(path.join(datadir, 'Snow_*PL200.mat')))
P400SET = dataset-P200SET

def mat_data(filename):
    dat = loadmat(filename)
    ind = next(i for i, x in enumerate(dat.keys()) if 'Snow_' in x)
    data = list(dat.values())[ind][0]
    keys = data.dtype.names
    values = data[0]
    return dict(zip(keys, values))

def mat2series(filename, key='azs'):
    data = mat_data(filename)
    values = data[key].flatten()
    t = list(map(mldatenum2datetime, data['time'].flatten()))
    series = pd.Series(data=values, index=t)
    series.name = key
    return series

def load_series(datafiles=P400SET, **kws):
    azss = []
    for fn in datafiles:
        try:
            azss.append(mat2series(fn, **kws))
        except KeyError:
            pass
    return pd.concat(azss).sort_index()

if __name__ == '__main__':
    data = load_series()
