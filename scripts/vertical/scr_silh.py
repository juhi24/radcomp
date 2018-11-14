# coding: utf-8
"""silhouette score analysis script"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification
from j24.tools import notify
import conf


def silh_score_avgs(cc, n_iter=10, vpc_conf=conf.VPC_PARAMS_SNOW, **kws):
    scores = pd.DataFrame()
    vconf = vpc_conf.copy()
    vconf.pop('n_clusters')
    for i in range(n_iter):
        for n_classes in range(5, 25):
            scheme = classification.VPC(n_clusters=n_classes, **vconf)
            cc.class_scheme = scheme
            cc.train(quiet=True)
            cc.classify()
            #fig, ax = plt.subplots()
            #cc.plot_silhouette(ax=ax)
            scores.loc[i, n_classes] = cc.silhouette_score(**kws)
    return scores


def plot_scores(scores, ax=None, **kws):
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
    scores = silh_score_avgs(cc, n_iter=10, **kws)
    notify('Silhouette score', 'Score calculation finished.')
    fig, ax = plt.subplots(dpi=110, figsize=(5,4))
    plot_scores(scores, ax=ax)
    return fig, ax


if __name__ == '__main__':
    scheme_id = conf.SCHEME_ID_MELT
    vpc_conf = conf.VPC_PARAMS_RAIN
    plt.close('all')
    cases = conf.init_cases(season='snow')
    cc = multicase.MultiCase.by_combining(cases, has_ml=False)
    del(cases)
    #fig, ax = score_analysis(cc, cols=(0, 1, 2, 'temp_mean'), weights=(1,1,1,0.8), vpc_conf=vpc_conf)
    #fig, ax = score_analysis(cc, cols=(0, 1, 2), weights=(1,1,1), vpc_conf=vpc_conf)
    fig, ax = score_analysis(cc, cols='all', vpc_conf=vpc_conf)
    fname = 'silh_score_{}.svg'.format(vpc_conf['basename'])
    savefile = path.join(conf.P1_FIG_DIR, fname)
    fig.savefig(savefile, bbox_inches='tight')
