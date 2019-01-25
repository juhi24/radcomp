# coding: utf-8
"""plot wrappers"""

from os import path

from radcomp.vertical import RESULTS_DIR

from j24 import ensure_join


SAVE_DEFAULT = True
SAVE_KWS = dict(bbox_inches='tight')


def subdir_vpc(c, subdir):
    name = c.vpc.name()
    return ensure_join(RESULTS_DIR, subdir, name)


def plot_cluster_centroids(c, save=SAVE_DEFAULT, **kws):
    """plot_cluster_centroids wrapper"""
    fig, axarr, _ = c.plot_cluster_centroids(fig_scale_factor=0.8)
    if save:
        savedir = subdir_vpc(c, 'classes_summary')
        fig.savefig(path.join(savedir, 'centroids.png'), **SAVE_KWS)


def plot_silhouette():
    pass

