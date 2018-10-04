# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pyart.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from glob import glob
from radcomp.vertical import RESULTS_DIR


def lonlat2csv(lon, lat, path_or_buf=None):
    """Save arrays of longitude and latitude to csv."""
    lonlat = pd.DataFrame(np.array([lon, lat]).T)
    path_or_buf = path_or_buf or path.join(RESULTS_DIR, 'ika_rhi_latlon.csv')
    lonlat.to_csv(path_or_buf, index=False, header=False)


if __name__ == '__main__':
    plt.close('all')
    lon_target = 24.28878
    datadir = path.expanduser('~/DATA/IKA/20180818RHI')
    datadir = path.expanduser('~/DATA/IKA/test')
    filefmt = '*RHI*.raw'
    filesglob = path.join(datadir, filefmt)
    files = np.sort(glob(filesglob))
    radar = pyart.io.read(files[0])
    z = radar.fields['reflectivity']['data']
    low_lat = radar.gate_latitude['data'][0]
    low_alt = radar.gate_altitude['data'][0]
    low_lon = radar.gate_longitude['data'][0]

