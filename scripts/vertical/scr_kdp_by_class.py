# coding: utf-8
"""Visualize kdp vs t_s by class"""
from os import path

import numpy as np
import matplotlib.pyplot as plt

from radcomp.visualization import LABELS
from radcomp.vertical import RESULTS_DIR
from j24.visualization import scatter_kde
from j24 import ensure_join


def hlines(ax):
    """hlines at important temperatures"""
    ax.axhline(-3, color='black')
    ax.axhline(-8, color='black')
    ax.axhline(-10, color='black')
    ax.axhline(-20, color='black')


def scatter_kdp(kdp, t, ax=None):
    ax = ax or plt.gca()
    selection = ~np.isnan(kdp.values.flatten())
    kdp_flat = kdp.values.flatten()[selection]
    t_flat = t.values.flatten()[selection]
    scatter_kde(kdp_flat, t_flat, ax=ax)
    ax.set_xlim(left=0, right=0.2)
    ax.set_ylim(bottom=-35, top=0)
    ax.invert_yaxis()
    ax.set_ylabel(LABELS['T'])
    ax.set_xlabel(LABELS['KDP'])
    hlines(ax)
    return fig, ax


def scatter_kdpg(kdpg, t, ax=None):
    ax = ax or plt.gca()
    selection = ~np.isnan(kdpg.values.flatten())
    kdpg_flat = kdpg.values.flatten()[selection]
    t_flat = t.values.flatten()[selection]
    scatter_kde(kdpg_flat, t_flat, ax=ax)
    ax.set_xlim(left=-2, right=6)
    ax.set_ylim(bottom=-35, top=0)
    ax.invert_yaxis()
    ax.set_ylabel(LABELS['T'])
    ax.set_xlabel(LABELS['KDPG'])
    ax.axvline(0, color='black')
    hlines(ax)
    return ax


if __name__ == '__main__':
    plt.ioff()
    for cl in range(cc.vpc.n_clusters):
        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
        t = cc.data['T'].loc[:, cc.classes()==cl]
        kdp = cc.data.kdp.loc[:, cc.classes()==cl]
        kdpg = cc.data.kdpg.loc[:, cc.classes()==cl]
        scatter_kdp(kdp, t, ax=axarr[0])
        title = 'Class {}'.format(cl)
        scatter_kdpg(kdpg, t, ax=axarr[1])
        fig.suptitle(title)
        outdir = ensure_join(RESULTS_DIR, 'kdp-t_scatter')
        outfile = path.join(outdir, 'cl{}.png'.format(cl))
        fig.savefig(outfile, bbox_inches='tight')
        plt.close(fig)