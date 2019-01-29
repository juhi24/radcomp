# coding: utf-8
"""plot wrappers"""

from os import path

import matplotlib.pyplot as plt

from radcomp.vertical import plotting, RESULTS_DIR

from j24 import ensure_join


SAVE_DEFAULT = True
SAVE_KWS = dict(bbox_inches='tight')


def subdir_vpc(vpc, subdir):
    name = vpc.name()
    return ensure_join(RESULTS_DIR, subdir, name)


def plot_cluster_centroids(vpc, save=SAVE_DEFAULT, **kws):
    """plot_cluster_centroids wrapper"""
    fig, axarr, _ = vpc.plot_cluster_centroids(fig_scale_factor=0.8)
    if save:
        savedir = subdir_vpc(vpc, 'classes_summary')
        fig.savefig(path.join(savedir, 'centroids.png'), **SAVE_KWS)


def plot_silhouette():
    return


if __name__ == '__main__':
    plt.close('all')
    fig, ax, bp_top = plotting.boxplot_t_combined(c, i_dis=range(5))