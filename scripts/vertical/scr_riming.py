# coding: utf-8

import pandas as pd
from glob import glob
from datetime import datetime, timedelta
from os import path
from scipy.io import loadmat
from j24 import home

DATAPATH = path.join(home(), 'DATA')

hdfpath = path.join(DATAPATH, 'FR_haoran.h5')

def path_dstr(fpath, start_char=4):
    fname = path.basename(fpath)
    return fname[start_char:start_char+8]

def parse_date_from_fpath(fpath, **kws):
    dstr = path_dstr(fpath, **kws)
    return datetime.strptime(dstr, '%Y%m%d')

def times_from_file(fpath, **kws):
    date = parse_date_from_fpath(fpath, **kws)
    data = loadmat(fpath)
    h = data['PIP_time'].flatten()
    return list(map(lambda hh: date+timedelta(hours=hh), h))

def read_data(dstr):
    ar_file = glob(path.join(DATAPATH, 'AR', 'ALL_{}*.mat'.format(dstr)))[0]
    fr_file = glob(path.join(DATAPATH, 'FR', 'FR_{}*.mat'.format(dstr)))[0]
    fr_data = loadmat(fr_file)['FR'][0]
    t = times_from_file(ar_file)
    return pd.Series(data=fr_data, index=t, name='FR')

if __name__ == '__main__':
    fr_files = glob(path.join(DATAPATH, 'FR', 'FR_*.mat'))
    fr_dstrs = map(lambda x:path_dstr(x, start_char=3), fr_files)
    datas = list(map(read_data, fr_dstrs))
    fr = pd.concat(datas)
    fr.sort_index(inplace=True)
    fr.to_hdf(hdfpath, 'data', mode='w')

