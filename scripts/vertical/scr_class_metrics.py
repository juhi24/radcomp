# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from radcomp.vertical import multicase, classification
from conf import SCHEME_ID_MELT, CASES_MELT


case_set = CASES_MELT
scheme_id = SCHEME_ID_MELT


basename = 'mlt'
params = ['ZH', 'zdr', 'kdp']
hlimits = (190, 10e3)
n_eigens = 15
reduced = True
use_temperature = False
radar_weight_factors = dict()


def silh_score_avgs(cc, n_iter=10, **kws):
    scores = pd.DataFrame()
    for i in range(n_iter):
        for n_classes in range(5, 25):
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


if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases(case_set)
    cases = cases[cases.ml_ok.astype(bool)]
    cc = multicase.MultiCase.by_combining(cases, has_ml=True)
    cc.load_classification(scheme_id)
    #scores = silh_score_avgs(cc, n_iter=5, n_pc=4)
    #scores.mean().plot()
    #cases.case.apply(lambda x: x.load_classification(scheme_id))
    #c = cases.case.iloc[0]
    order = cc.clus_centroids()[0].ZH.iloc[0]
    cc.plot_cluster_centroids(cmap='viridis', colorful_bars='blue', sortby=order)
    #fig, ax = plt.subplots()
    #cc.plot_silhouette(ax=ax)


