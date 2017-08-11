# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR, cloud_top_h
from j24 import ensure_dir

save = True
plot_by_class = False

plt.ion()
#plt.close('all')
np.random.seed(0)
cmap = 'gist_ncar'
colorful_bars = 'blue'

n_eigens = 25
n_clusters = 25
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

def fr_comb(cases):
    # TODO: duplicate in scr_class_scatter
    fr_list = []
    for cname in cases.index:
        #data_g = g.loc[name]
        fr_list.append(cases.case[cname].fr())
    return pd.concat(fr_list)

def classes_comb(cases):
    classes_list = []
    for cname in cases.index:
        c = cases.case[cname]
        c.load_classification(name)
        classes = c.classes
        classes.index = classes.index.round('1min')
        classes.name = 'class'
        classes_list.append(classes)
    return pd.concat(classes_list)

def fr_grouped(cases):
    fr = fr_comb(cases)
    classes = classes_comb(cases)
    return fr.groupby(by=classes)

def consec_occ_group(classes):
    consec_count_g = classes.groupby((classes != classes.shift()).cumsum())
    count = consec_count_g.count()
    count.name = 'count'
    consec_count = pd.concat([consec_count_g.mean(), count], axis=1)
    consec_count.index.name = 'id'
    return consec_count.groupby(by='class')

cases = case.read_cases('14-16by_hand')
name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary', name))
c = case.Case.by_combining(cases)
c.load_classification(name)
print(consec_occ_group(c.classes).mean())
z = c.clus_centroids()[0].loc['ZH']
toph = cloud_top_h(z)
fr_med = fr_grouped(cases).median()
n_classes = c.classes.unique().size
fr_med = fr_med.reindex(index=range(n_classes))

if plot_by_class:
    df = c.pcolor_classes(cmap=cmap)
f_cen, axarr_cen, order = c.plot_cluster_centroids(cmap=cmap, sortby='temp_mean',
                                            colorful_bars=colorful_bars)
c.plot_cluster_centroids(cmap=cmap, sortby=toph, colorful_bars=colorful_bars)
f, axfr, order = c.plot_cluster_centroids(cmap=cmap, sortby=fr_med,
                                        colorful_bars=colorful_bars,
                                        n_extra_ax=1)
ax=axfr[3]
fr_med.reindex(index=order).plot.bar(ax=ax)

if save:
    if plot_by_class:
        for i, fig in df.fig.iteritems():
            fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
    f_cen.savefig(path.join(results_dir, 'centroids.png'))


