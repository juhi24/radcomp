# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, classification
from j24.tools import notify
from conf import SCHEME_ID_MELT, SCHEME_ID_SNOW, CASES_MELT, CASES_SNOW, P1_FIG_DIR


case_set = CASES_SNOW
scheme_id = SCHEME_ID_SNOW


basename = 'mlt2'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 18
reduced = True
use_temperature = False
radar_weight_factors = dict()


def silh_score_avgs(cc, n_iter=10, **kws):
    scores = pd.DataFrame()
    for i in range(n_iter):
        for n_classes in range(8, 25):
            scheme = classification.VPC(params=params, hlimits=hlimits,
                                        n_eigens=n_eigens, reduced=reduced,
                                        radar_weight_factors=radar_weight_factors,
                                        basename=basename, n_clusters=n_classes)
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


def score_analysis(cc):
    """Compute and plot silhouette scores per number of classes."""
    scores = silh_score_avgs(cc, n_iter=10, n_pc=3)
    notify('Silhouette score', 'Score calculation finished.')
    fig, ax = plt.subplots(dpi=110, figsize=(5,4))
    plot_scores(scores, ax=ax)
    savefile = path.join(P1_FIG_DIR, 'silh_score.svg')
    fig.savefig(savefile, bbox_inches='tight')


if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases(case_set)
    #cases = cases[cases.ml_ok.astype(bool)]
    cc = multicase.MultiCase.by_combining(cases, has_ml=False)
    for _, c in cases.case.iteritems():
        c.load_classification(scheme_id)
