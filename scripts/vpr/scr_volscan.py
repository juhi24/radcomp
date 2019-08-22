# coding: utf-8

from os import path
from glob import glob

#import pandas as pd
import pyart
import numpy as np
import matplotlib.pyplot as plt


HYDE_AZIMUTH = 81.89208


def create_volume_scan(files):
    """volume scan from multiple radar data files"""
    r = None
    for f in files:
        if r is None:
            r = pyart.io.read(f)
            continue
        r = pyart.util.join_radar(r, pyart.io.read(f))
    return r


def plot_vrhi_vs_rhi(vrhi, rhi, field='reflectivity'):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    vdisp = pyart.graph.RadarDisplay(vrhi)
    rhidisp = pyart.graph.RadarDisplay(rhi)
    vkw = dict(vmin=-10, vmax=40)
    vdisp.plot(field, ax=axarr[0], **vkw)
    rhidisp.plot(field, ax=axarr[1], **vkw)
    for ax in axarr:
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 80)
    return axarr


if __name__ == '__main__':
    hdd = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
    rhifile = path.join(hdd, 'IKA_final/20140221/201402212355_IKA.RHI_HV.raw')
    rhi = pyart.io.read(rhifile)
    testdir = path.expanduser(path.join(hdd, 'test_volscan2'))
    files = glob(path.join(testdir, '*[A-F].raw'))
    files.sort()
    vr = create_volume_scan(files)
    vrhi = pyart.util.cross_section_ppi(vr, [HYDE_AZIMUTH])
    axarr = plot_vrhi_vs_rhi(vrhi, rhi)

