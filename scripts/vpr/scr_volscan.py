# coding: utf-8

from os import path
from glob import glob

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import radcomp.visualization as vis
from radcomp.tools import rhi

from j24 import ensure_dir


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
    vr = rhi.create_volume_scan(files)
    vrhi = pyart.util.cross_section_ppi(vr, [rhi.AZIM_IKA_HYDE])
    #axarr = plot_vrhi_vs_rhi(vrhi, rrhi, field='KDP')
    #filedir = path.join(hdd, 'test_volscan3')
#    filedir = path.join(hdd, 'IKA_final', '20140221')
#    outdir = ensure_dir(path.expanduser('~/DATA/vpvol'))
#    ds = rhi.xarray_workflow(filedir, dir_out=None, recalculate_kdp=False)
#    plt.figure()
#    ds.KDP.T.plot()
    kdp = pyart.retrieve.kdp_maesaka(vrhi, Clpf=5000)[0]
    kdp['data'] = np.ma.masked_array(data=kdp['data'], mask=vrhi.fields['differential_phase']['data'].mask)
    vrhi.fields['kdp'] = kdp
    fig, axarr = plt.subplots(ncols=2, figsize=(12,5))
    disp = pyart.graph.RadarDisplay(vrhi)
    disp.plot('specific_differential_phase', vmin=0, vmax=0.3, ax=axarr[0], cmap='viridis')
    disp.plot('kdp', vmin=0, vmax=0.3, ax=axarr[1])

