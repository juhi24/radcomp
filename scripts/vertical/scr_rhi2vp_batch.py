# coding: utf-8

import pandas as pd
from os import path
from datetime import datetime
from radcomp.tools.rhi import rhi2vp


DATE_FMT = '%Y%m%d%H%M'


def parse_file_date(basename):
    datestr = basename[:12]
    return datetime.strptime(datestr, DATE_FMT)


def read_filelist(listfile):
    files = pd.read_csv(listfile, header=None, names=['relpath'])
    files['basename'] = files.relpath.apply(path.basename)
    files.drop_duplicates('basename', inplace=True)
    files['dirname'] = files.relpath.apply(path.dirname)
    files['time'] = files.basename.apply(parse_file_date)
    files.set_index(files.time, inplace=True)
    return files.drop('time', axis=1)


if __name__ == '__main__':
    home = path.expanduser('~')
    datapath = path.join(home, 'mnt')
    resultsdir = path.join(home, 'DATA1', 'vprhi')
    listfile = path.join(home, 'ika_rhi.list')
    files = read_filelist(listfile)
    for dirname in files.dirname.unique():
        datadir = path.join(datapath, dirname)
        rhi2vp(datadir, resultsdir)

