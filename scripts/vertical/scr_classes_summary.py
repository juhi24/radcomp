# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, casedf, classification, RESULTS_DIR, echo_top_h
from j24 import ensure_dir

save = False
plot_by_class = False

plt.ion()
#plt.close('all')
cmap = 'pyart_RefDiff'
colorful_bars = 'blue'

n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

def order2pos(order):
    return pd.Series(order).sort_values().index.values

def var_grouped(cases, func):
    fr = func(cases)
    classes = casedf.classes_comb(cases, name)
    return fr.groupby(by=classes)

def var_cla(cases, name, func):
    return pd.concat([casedf.classes_comb(cases, name), func(cases)], axis=1)

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
toph = echo_top_h(z)
fr_med = var_grouped(cases, casedf.fr_comb).median()
n_classes = c.classes.unique().size
fr_med = fr_med.reindex(index=range(n_classes))

if plot_by_class:
    df = c.pcolor_classes(cmap=cmap)
f_t, ax_t, order_t = c.plot_cluster_centroids(cmap=cmap, sortby='temp_mean',
                                              colorful_bars=colorful_bars,
                                              n_extra_ax=1)
f_fr, ax_fr, order_fr = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                                 colorful_bars=colorful_bars,
                                                 n_extra_ax=1)
f_fr, ax_fr, order_fr = c.plot_cluster_centroids(cmap=cmap, sortby=fr_med,
                                                 colorful_bars=colorful_bars,
                                                 n_extra_ax=1)
ax=ax_t[3]
#fr_med.reindex(index=order).plot.bar(ax=ax)
t_cla = var_cla(cases, name, casedf.t_comb)
t_cla.boxplot(by='class', positions=order2pos(order_t), ax=ax)
ax.set_title('')
f_t.suptitle('')

if save:
    if plot_by_class:
        for i, fig in df.fig.iteritems():
            fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
    f_t.savefig(path.join(results_dir, 'centroids.png'))


