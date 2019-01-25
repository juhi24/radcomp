# coding: utf-8

import pyart
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import RESULTS_DIR
from j24 import home


if __name__ == '__main__':
    plt.close('all')
    rawfile = path.join(home(), 'DATA', '201402212355_IKA.RHI_HV.raw')
    outfile = path.join(RESULTS_DIR, 'rhi_sample.png')
    radar = pyart.io.read(rawfile)
    fig, ax = plt.subplots()
    display = pyart.graph.RadarDisplay(radar)
    display.plot('reflectivity')
    ax.set_ylim(bottom=0, top=10)
    ax.set_xlim(left=0, right=80)
    fig.savefig(outfile)
