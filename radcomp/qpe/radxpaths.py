#!/usr/bin/env python2
# coding: utf-8
import copy
import netCDF4 as nc
from os import path
from glob import glob


def data_is_bad(ncdata):
    lvar = list(ncdata.variables)
    z = ncdata.variables['z0']
    not_ppi = False # TODO
    no_kdp = 'KDP' not in lvar # no KDP
    high_elev = z[0] > 1.5
    low_elev = z[0] < 0.05
    is_correct_van_elev = int(round(z[0]*10)) == 7
    van_wrong_elev = ncdata.title == 'VANTAA' and not is_correct_van_elev
    weird_kum = 'kum' in ncdata.title and not ('UNKNOWN_ID_73' in lvar or 'UNKNOWN_ID_74' in lvar)
    return no_kdp or high_elev or not_ppi or van_wrong_elev or low_elev or weird_kum


def filter_filepaths(filepaths_all):
    filepaths_good = copy.deepcopy(filepaths_all)
    for filepath in filepaths_all:
        with nc.Dataset(filepath, 'r') as ncdata:
            if data_is_bad(ncdata):
                #print('bad: ' + filepath)
                filepaths_good.remove(filepath)
    return filepaths_good


def fpath(site):
    return path.join(GRIDPATH, '{}_goodfiles.csv'.format(site))


def save():
    filepaths_all = glob(path.join(GRIDPATH, '???', '*', 'ncf_20160903_[12]?????.nc'))
    filepaths_all.extend(glob(path.join(GRIDPATH, '???', '*', 'ncf_20160904_0[0-6]????.nc')))
    filepaths_all.sort()
    filepaths_good = filter_filepaths(filepaths_all) # takes a lot of time!
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
