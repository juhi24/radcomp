# coding: utf-8

import glob
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from scipy.io import savemat
from radcomp import radx

basepath = path.join('/media', getpass.getuser(), '04fafa8f-c3ca-48ee-ae7f-046cf576b1ee')
GRIDPATH = path.join(basepath, 'grid')

def fpath(site):
    return path.join(GRIDPATH, '{}_goodfiles.csv'.format(site))

def save():
    filepaths_all = glob.glob(path.join(GRIDPATH, '???', '*', 'ncf_20160903_[12]?????.nc'))
    filepaths_all.extend(glob.glob(path.join(GRIDPATH, '???', '*', 'ncf_20160904_0[0-6]????.nc')))
    filepaths_all.sort()
    filepaths_good = radx.filter_filepaths(filepaths_all) # takes a lot of time!
    for site in radx.SITES:
        paths = [k for k in filepaths_good if '/{}/'.format(site) in k]
        outpath = fpath(site)
        spaths = pd.Series(paths)
        spaths.to_csv(path=outpath, index=False)

def load():
    out = dict(KUM=None, KER=None, VAN=None)
    for site in out:
        out[site] = pd.read_csv(fpath(site), header=None, names=['filepath'])
    return out


d = load()
dran = {}
for site in radx.SITES:
    f = d[site].iloc[0].values[0]
    rad = radx.RADXgrid(f)
    ran = rad.data.variables['lat0'][:].copy()
    for (x,y), element in np.ndenumerate(ran):
        ran[y, x] = rad.distance_from_radar(x, y)
    dran['range_' + site.lower()] = ran
    if False:
        plt.figure()
        plt.pcolormesh(ran)
        plt.title(site)
ran_mat_path = path.join(GRIDPATH, 'range.mat')
savemat(ran_mat_path, dran)
