# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

from radcomp import azs


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    data = azs.load_series()
    types = {'class': int, 'azs': float}
    azs_cl = pd.concat([cc_s.classes(), data], axis=1).dropna().astype(types)
    fig, axarr, _ = cc_r.plot_cluster_centroids(fields=['zh'], plot_counts=False, n_extra_ax=2)
    azs_cl.boxplot(by='class', ax=axarr[-2])
    g = azs_cl.groupby('class')
    g.size().plot.bar(ax=axarr[-1])


