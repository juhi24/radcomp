# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from radcomp.vertical import case, classification


case_set = 'melting'
scheme_id = 'mlt_20eig15clus_pca'


if __name__ == '__main__':
    cases = case.read_cases(case_set)
    cases = cases[cases.ml_ok.astype(bool)]
    plt.close('all')
    #cases.case.apply(lambda x: x.load_classification(scheme_id))
    #c = cases.case.iloc[0]
    scheme = classification.load(scheme_id)
    cc = case.Case.by_combining(cases, class_scheme=scheme, has_ml=True)
    cc.load_classification(scheme_id)
    order = cc.clus_centroids()[0].ZH.iloc[0]
    #c.plot_cluster_centroids(cmap='viridis', colorful_bars='blue', sortby=order)
    sh = cc.silhouette_coef()

