# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pyart.io
import matplotlib.pyplot as plt
from os import path
from glob import glob
from scr_rhi2vp import R_IKA_HYDE


if __name__ == '__main__':
    lon_target = 24.28878
    #datadir = path.expanduser('~/DATA/IKA/20180818RHI')
    datadir = path.expanduser('~/DATA/IKA/test')
    filefmt = '*RHI*.raw'
    filesglob = path.join(datadir, filefmt)
    files = glob(filesglob)
    radar = pyart.io.read(files[0])
    z = radar.fields['reflectivity']['data']
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_below('cross_correlation_ratio', 0.8)
    kdp_m=pyart.retrieve.kdp_maesaka(radar)
    #kdp_s=pyart.retrieve.kdp_schneebeli(radar)
    #kdp_v=pyart.retrieve.kdp_vulpiani(radar)
    radar.add_field('kdp_maesaka', kdp_m[0])
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    display = pyart.graph.RadarDisplay(radar)
    display.plot_rhi('kdp_maesaka', vmin=0, vmax=0.2, ax=ax[0], cmap='viridis')
    display.plot_rhi('specific_differential_phase', cmap='viridis', vmin=0, vmax=0.2, ax=ax[1])
    r_km = R_IKA_HYDE/1000
    margin = 1
    for axi in ax:
        axi.axvline(r_km+margin)
        axi.axvline(r_km-margin)
        axi.axvline(r_km, color='gray', alpha=0.2)


