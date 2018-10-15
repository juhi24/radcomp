# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
import pyart.io
from os import path


if __name__ == '__main__':
    filepath = path.expanduser('~/DATA/vprhi2/201504221810_IKA.RHI_HV.raw')
    radar = pyart.io.read(filepath)
    kdp_m = pyart.retrieve.kdp_maesaka(radar)#, check_outliers=False)
    radar.add_field('kdp_maesaka', kdp_m[0])
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    display = pyart.graph.RadarDisplay(radar)
    display.plot_rhi('kdp_maesaka', vmin=0, vmax=0.2, ax=ax[0], cmap='viridis')
    display.plot_rhi('specific_differential_phase', cmap='viridis', vmin=0, vmax=0.2, ax=ax[1])
    