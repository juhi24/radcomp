# coding: utf-8

import os

from radcomp.tools import rhi

if __name__ == '__main__':
    datadir = os.path.expanduser('~/datatmp/data')
    dir_in = os.path.join(datadir, 'IKA_final/20140222')
    dir_out = os.path.join(datadir, 'vrhi3')
    os.makedirs(dir_out, exist_ok=True)
    rhi.nc_workflow(dir_in, dir_out)