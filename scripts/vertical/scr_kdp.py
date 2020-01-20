# coding: utf-8
"""experiment kdp recalculation"""

import pyart.io
import matplotlib.pyplot as plt
from os import path
from glob import glob
from radcomp.tools import rhi


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
    radar = rhi.kdp_csu(radar)

    opt = dict(psidp_field='FDP')
    kdp_m=pyart.retrieve.kdp_maesaka(radar)
    kdp_s=pyart.retrieve.kdp_schneebeli(radar, **opt)
    kdp_v=pyart.retrieve.kdp_vulpiani(radar, **opt)
    radar.add_field('kdp_maesaka', kdp_m[0])
    radar.add_field('kdp_s', kdp_s[0])
    radar.add_field('kdp_v', kdp_v[0])

    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    display = pyart.graph.RadarDisplay(radar)
    kws = dict(vmin=-0.05, vmax=0.2, cmap='viridis')
    dpkws = dict(vmin=100, vmax=150, cmap='cubehelix')
    display.plot_rhi('differential_phase', ax=ax[0,0], **dpkws)
    display.plot_rhi('kdp_csu', ax=ax[0,2], **kws)
    display.plot_rhi('kdp_maesaka', ax=ax[1,1], **kws)
    display.plot_rhi('kdp_s', ax=ax[1,3], **kws)
    display.plot_rhi('FDP', ax=ax[1,0], **dpkws)
    display.plot_rhi('kdp_v', ax=ax[1,2], **kws)
    display.plot_rhi('specific_differential_phase', ax=ax[0,1], **kws)
    r_km = rhi.R_IKA_HYDE/1000
    margin = 1
    for axi in ax.flatten():
        axi.axvline(r_km+margin)
        axi.axvline(r_km-margin)
        axi.axvline(r_km, color='gray', alpha=0.2)
        axi.set_ylim(0,6)
    plt.tight_layout()


