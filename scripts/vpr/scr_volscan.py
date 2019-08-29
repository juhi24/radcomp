# coding: utf-8

from os import path
from glob import glob

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import radcomp.visualization as vis
from radcomp.tools import rhi


AZIM_IKA_HYDE = 81.89208


def create_volume_scan(files):
    """volume scan from multiple radar data files"""
    r = None
    for f in files:
        if r is None:
            r = pyart.io.read(f)
            continue
        r = pyart.util.join_radar(r, pyart.io.read(f))
    return r


def plot_vrhi_vs_rhi(vrhi, rhi, field='ZH'):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    vdisp = pyart.graph.RadarDisplay(vrhi)
    rhidisp = pyart.graph.RadarDisplay(rhi)
    vkw = dict(vmin=vis.VMINS[field], vmax=vis.VMAXS[field])
    mapping = dict(ZH='reflectivity',
                   KDP='specific_differential_phase',
                   ZDR='differential reflectivity')
    if field in mapping:
        field = mapping[field]
    vdisp.plot(field, ax=axarr[0], **vkw)
    rhidisp.plot(field, ax=axarr[1], **vkw)
    for ax in axarr:
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 80)
    return axarr


if __name__ == '__main__':
    hdd = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee'
    rhifile = path.join(hdd, 'IKA_final/20140221/201402212355_IKA.RHI_HV.raw')
    rrhi = pyart.io.read(rhifile)
    testdir = path.expanduser(path.join(hdd, 'test_volscan2'))
    files = glob(path.join(testdir, '*[A-F].raw'))
    files.sort()
    vr = create_volume_scan(files)
    vrhi = pyart.util.cross_section_ppi(vr, [AZIM_IKA_HYDE])
    #axarr = plot_vrhi_vs_rhi(vrhi, rrhi, field='KDP')
    filedir = path.join(hdd, 'test_volscan3')
    fnames = pd.Series(glob(path.join(filedir, '*PPI3_[A-F].raw')))
    tstrs = fnames.apply(lambda s: s[-27:-15])
    g = fnames.groupby(tstrs)
    vrhis = dict()
    for tstr, df in g:
        print(tstr)
        df.sort_values(inplace=True)
        vs = create_volume_scan(df)
        vrhi = pyart.util.cross_section_ppi(vs, [AZIM_IKA_HYDE])
        t, vp = rhi.vrhi2vp(vrhi)
        vrhis[t] = vp
    df = pd.concat(vrhis)
    df.index.rename(['time', 'height'], inplace=True)
    ds = df.to_xarray()




