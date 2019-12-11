# coding: utf-8
""""vertical profile"""

# builtins
from os import path

# pypi
import pyart
import numpy as np
import pandas as pd
import xarray as xr

from radcomp.tools import rhi

if __name__ == '__main__':
    hdd = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
    filedir = path.join(hdd, 'IKA_final', '20140221')
    azims = np.arange(0, 360)
    kms = np.arange(50, 61)
    azim = 90
    km = 50
    g = rhi.volscan_groups(filedir)
    voldf = g.get_group('201402212355')
    voldf.sort_values(inplace=True)
    vs = rhi.create_volume_scan(voldf)
    dfs = []
    for azim in np.arange(0, 180, 10):
        vrhi = pyart.util.cross_section_ppi(vs, [azim])
        for km in kms:
            dist = km*1000
            rdr_vars, hght = rhi.rhi_preprocess(vrhi, r_poi=dist)
            df = rhi.agg2vp(hght, rdr_vars).drop('DP', axis=1)
            t = rhi.scan_timestamp(vrhi)
            df = pd.concat([df], keys=[dist], names=['range'])
            df = pd.concat([df], keys=[azim], names=['azimuth'])
            df = pd.concat([df], keys=[t], names=['time']) # prepend to
            dfs.append(df)
    ds = pd.concat(dfs).to_xarray()

