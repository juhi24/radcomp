# coding: utf-8
"""Visualize kdp vs t_s by class"""
import numpy as np
import matplotlib.pyplot as plt

from radcomp.visualization import LABELS
from j24.visualization import scatter_kde


if __name__ == '__main__':
    plt.ion()
    t12 = cc.data.T.loc[:, cc.classes()==12]
    kdp12 = cc.data.kdp.loc[:, cc.classes()==12]
    kdpg12 = cc.data.kdpg.loc[:, cc.classes()==12]
    selection = ~np.isnan(kdpg12.values.flatten())
    x = kdp12.values.flatten()[selection]
    y = t12.values.flatten()[selection]
    fig, ax = plt.subplots()
    scatter_kde(x, y, ax=ax)
    ax.set_xlim(left=0, right=0.2)
    ax.set_ylim(bottom=-35, top=0)
    ax.invert_yaxis()
    ax.set_ylabel(LABELS['T'])
    ax.set_xlabel(LABELS['KDP'])
