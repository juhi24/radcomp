# coding: utf-8
"""Visualize kdp vs t_s by class"""
import numpy as np
import matplotlib.pyplot as plt

from radcomp.visualization import LABELS
from j24.visualization import scatter_kde


def scatter_kdp(kdp, t):
    fig, ax = plt.subplots()
    selection = ~np.isnan(kdp.values.flatten())
    kdp_flat = kdp.values.flatten()[selection]
    t_flat = t.values.flatten()[selection]
    scatter_kde(kdp_flat, t_flat, ax=ax)
    ax.set_xlim(left=0, right=0.2)
    ax.set_ylim(bottom=-35, top=0)
    ax.invert_yaxis()
    ax.set_ylabel(LABELS['T'])
    ax.set_xlabel(LABELS['KDP'])
    return ax


def scatter_kdpg(kdpg, t):
    fig, ax = plt.subplots()
    selection = ~np.isnan(kdpg.values.flatten())
    kdpg_flat = kdpg.values.flatten()[selection]
    t_flat = t.values.flatten()[selection]
    scatter_kde(kdpg_flat, t_flat, ax=ax)
    ax.set_xlim(left=-2, right=6)
    ax.set_ylim(bottom=-35, top=0)
    ax.invert_yaxis()
    ax.set_ylabel(LABELS['T'])
    ax.set_xlabel(LABELS['KDPG'])
    return ax


if __name__ == '__main__':
    plt.ion()
    for cl in range(cc.vpc.n_clusters):
        t = cc.data['T'].loc[:, cc.classes()==cl]
        kdp = cc.data.kdp.loc[:, cc.classes()==cl]
        kdpg = cc.data.kdpg.loc[:, cc.classes()==cl]
        ax = scatter_kdp(kdp, t)
        title = 'Class {}'.format(cl)
        ax.set_title(title)
        axg = scatter_kdpg(kdpg, t)
        axg.set_title(title)