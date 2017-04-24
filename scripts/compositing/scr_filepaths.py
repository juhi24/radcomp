# coding: utf-8

import glob
import getpass
import pandas as pd
from os import path
from radcomp import radx

basepath = path.join('/media', getpass.getuser(), '04fafa8f-c3ca-48ee-ae7f-046cf576b1ee')
gridpath = path.join(basepath, 'grid')
filepaths_all = glob.glob(path.join(gridpath, '???', '*', 'ncf_20160903_[12]?????.nc'))
filepaths_all.extend(glob.glob(path.join(gridpath, '???', '*', 'ncf_20160904_0[0-6]????.nc')))
filepaths_all.sort()
filepaths_good = radx.filter_filepaths(filepaths_all) # takes a lot of time!
for site in radx.SITES:
    paths = [k for k in filepaths_good if '/{}/'.format(site) in k]
    csvpath = path.join(gridpath, '{}_goodfiles.csv'.format(site))
    pd.Series(paths).to_csv(path=csvpath, index=False)
