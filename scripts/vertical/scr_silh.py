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


vpc_conf_snow = dict(basename = 'snow',
                     params=['ZH', 'zdr', 'kdp'],
                     hlimits=(190, 10e3),
                     n_eigens=22,
                     reduced=True,
                     use_temperature=True,
                     t_weight_factor=0.8,
                     radar_weight_factors=dict(zdr=0.9, kdp=1.4))

vpc_conf_rain = dict(basename = 'mlt2',
                     params=['ZH', 'zdr', 'kdp'],
                     hlimits=(190, 10e3),
                     n_eigens=16,
                     reduced=True,
                     use_temperature=False,
                     radar_weight_factors=dict())


def silh_score_avgs(cc, n_iter=10, vpc_conf=vpc_conf_snow, **kws):
    scores = pd.DataFrame()
    for i in range(n_iter):
        for n_classes in range(10, 27):
            scheme = classification.VPC(n_clusters=n_classes, **vpc_conf)
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
    vpc_conf = vpc_conf_rain
    plt.close('all')
    cases = conf.init_cases(season='snow')
    cc = multicase.MultiCase.by_combining(cases, has_ml=False)
    del(cases)
    fig, ax = score_analysis(cc, cols=(0, 1, 2), weights=(1,1,1), vpc_conf=vpc_conf)
    fname = 'silh_score_{}.svg'.format(vpc_conf['basename'])
    savefile = path.join(conf.P1_FIG_DIR, fname)
    fig.savefig(savefile, bbox_inches='tight')
