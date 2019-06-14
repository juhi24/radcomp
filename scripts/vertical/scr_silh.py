# coding: utf-8
"""silhouette score analysis script"""

from os import path
import copy

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from radcomp.vertical import multicase, classification
from j24.tools import notify

import conf


def silh_score_avgs(cc, n_iter=12, vpc_conf=conf.VPC_PARAMS_SNOW, **kws):
    """Compute silhouette scores per number of classes."""
    scores = pd.DataFrame()
    vconf = vpc_conf.copy()
    vconf.pop('n_clusters')
    for i in range(n_iter):
        for n_classes in range(5, 21):
            vpc = classification.VPC(n_clusters=n_classes, **vconf)
            cc.vpc = vpc
            cc.train(quiet=True)
            cc.classify()
            #fig, ax = plt.subplots()
            #cc.plot_silhouette(ax=ax)
            scores.loc[i, n_classes] = vpc.silhouette_score(**kws)
    return scores


def plot_scores(scores, ax=None, **kws):
    """Plot the silhouette score mean and standard deviation."""
    n_classes = scores.columns.values
    ax = ax or plt.gca()
    score = scores.mean()
    std = scores.std()
    score.plot(ax=ax, color='black', label='mean score')
    ax.fill_between(n_classes, score-std, score+std, alpha=0.3,
                    facecolor='blue', label='score std', **kws)
    ax.set_xlabel('Number of classes')
    ax.set_ylabel('Silhouette score')
    ax.set_xlim(left=n_classes[0], right=n_classes[-1])
    ax.legend()
    ax.grid(axis='x')
    return ax


def score_analysis(cc, **kws):
    """Compute and plot silhouette scores per number of classes."""
    scores = silh_score_avgs(cc, **kws)
    notify('Silhouette score', 'Score calculation finished.')
    fig, ax = plt.subplots(dpi=110, figsize=(5,4))
    plot_scores(scores, ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig, ax, scores


if __name__ == '__main__':
    plt.close('all')
    season = 'snow'
    vpc_conf = conf.VPC_PARAMS_RAIN if season=='rain' else conf.VPC_PARAMS_SNOW
    #cc = multicase.MultiCase.from_caselist(season, has_ml=(season == 'rain'))
    cc = copy.deepcopy(cc_s)
    #fig, ax = score_analysis(cc, cols=(0, 1, 2, 'temp_mean'), weights=(1,1,1,0.8), vpc_conf=vpc_conf)
    #fig, ax = score_analysis(cc, cols=(0, 1, 2), weights=(1,1,1), vpc_conf=vpc_conf)
    fig, ax, scores = score_analysis(cc, cols='all', vpc_conf=vpc_conf)
    fname = 'silh_score_{}.svg'.format(cc.vpc.name())
    savefile = path.join(conf.P1_FIG_DIR, fname)
    fig.savefig(savefile, bbox_inches='tight', dpi=300)
