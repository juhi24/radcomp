# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from radcomp.vertical import plotting

import plot_wrappers


if __name__ == '__main__':
    vpc = cc_r.vpc
    plt.close('all')
    split = 0.14
    gs_kws = dict(left_extra=split)
    #fig, _, _ = vpc.plot_cluster_centroids(gs_kws=gs_kws)
    fig, _ = plot_wrappers.plot_rain_case(cases_r, save=False,
                                          interactive=False, gs_kws=gs_kws)
    gs = plotting._gs_extra(1, 4, right=split*0.7)
    ax = fig.add_subplot(gs[1:3])
    plotting.cl_colorbar(vpc, ax, title='Class')
