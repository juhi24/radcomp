# coding: utf-8

import numpy as np
import pandas as pd
from scipy.io import loadmat
from os import path
from j24 import home, mldatenum2datetime

fpath = path.join(home(), 'DATA', 'FMI_meteo_data_2014_2017.mat')
hdfpath = path.join(home(), 'DATA', 't_fmi_14-17.h5')
data = loadmat(fpath)['meteo_fmi']

class TFMI:
    """FMI temperature mat struct"""
    def __init__(self, data):
        self.data = data # struct from mat

    def get(self, param):
        return data[param][0][0].flatten()

    def time(self):
        return np.array(list(map(mldatenum2datetime, self.get('ObsTime'))))

    def fields(self):
        return list(self.data[0].dtype.fields)

    def to_dataframe(self):
        f = self.fields() # copy
        f.remove('ObsTime')
        index = self.time()
        data = {}
        for field in f:
            data[field] = self.get(field)
        return pd.DataFrame(index=index, data=data)


t = TFMI(data)
df = t.to_dataframe()
df.to_hdf(hdfpath, 'data', mode='w')
