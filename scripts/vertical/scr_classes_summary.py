# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import pyart # to register colormaps
from os import path
from radcomp.vertical import (case, casedf, classification, plotting,
                              RESULTS_DIR, echo_top_h)
from j24 import ensure_dir

save = True
plot_by_class = False

plt.ion()
plt.close('all')
cmap = 'pyart_RefDiff'
colorful_bars = 'blue'

n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

def var2order(var):
    return var.sort_values().index.values

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

def boxplot(varcl, ax=None, sortby=None, **kws):
    """pandas boxplot wrapper"""
    if ax is None:
        ax = plt.gca()
    if sortby is None:
        positions = range(1, varcl.shape[0])
    else:
        positions = order2pos(var2order(sortby))
    varcl.boxplot(by='class', positions=positions, ax=ax, **kws)
    ax.set_title('')
    ax.figure.suptitle('')
    ax.set_xlabel('Class ID')
    plt.xticks(rotation=0)
    ax.xaxis.grid(False)

cases = case.read_cases('14-16by_hand')
name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary', name))
c = case.Case.by_combining(cases)
c.load_classification(name)
# TODO: below this line needs cleaning
print(consec_occ_group(c.classes).mean())
z = c.clus_centroids()[0].loc['ZH']
toph = echo_top_h(z)
fr_med = var_grouped(cases, casedf.fr_comb).median()
n_classes = c.classes.unique().size
fr_med = fr_med.reindex(index=range(n_classes))

f_t, ax_t, order_t = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                              colorful_bars=colorful_bars,
                                              n_extra_ax=0, plot_counts=True)
f_lwe, ax_lwe, order_lwe = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                                    colorful_bars=colorful_bars,
                                                    n_extra_ax=1, plot_counts=False)
f_fr, ax_fr, order_fr = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                                 colorful_bars=colorful_bars,
                                                 n_extra_ax=1, plot_counts=False)
f_azs, ax_azs, order_azs = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                                 colorful_bars=colorful_bars,
                                                 n_extra_ax=1, plot_counts=False)
f_n, ax_n, order_n = c.plot_cluster_centroids(cmap=cmap, sortby=toph,
                                              colorful_bars=colorful_bars,
                                              plot_counts=True)
#fr_med.reindex(index=order).plot.bar(ax=ax)
ib = -1
axb_t = ax_t[-2]
axb_lwe = ax_lwe[ib]
axb_fr = ax_fr[ib]
axb_azs = ax_azs[ib]
t_cla = var_cla(cases, name, casedf.t_comb)
lwe_cla = var_cla(cases, name, casedf.lwe_comb)
fr_cla = var_cla(cases, name, casedf.fr_comb)
azs_cla = var_cla(cases, name, casedf.azs_comb)
boxplot(t_cla, ax=axb_t, sortby=toph)
boxplot(lwe_cla, ax=axb_lwe, sortby=toph)
boxplot(fr_cla, ax=axb_fr, sortby=toph)
boxplot(azs_cla, ax=axb_azs, sortby=toph)
ax_t[-2].set_ylabel('$T_{cen}$, $^{\circ}C$')
axb_t.set_ylabel(plotting.LABELS['temp_mean'])
axb_lwe.set_ylabel(plotting.LABELS['intensity'])
axb_fr.set_ylabel(plotting.LABELS['FR'])
axb_azs.set_ylabel('$\\alpha_{ZS}$')
axb_azs.set_yscale('log')
for ax in (ax_t[0], ax_lwe[0], ax_fr[0], ax_n[0], ax_azs[0]):
    ax.set_title('Cluster centroids by cloud top height')
ax_t[-1].set_ylim(bottom=0, top=450)
plt.xticks(rotation=0)

if save:
    savekws = dict(bbox_inches='tight')
    if plot_by_class:
        for i, fig in df.fig.iteritems():
            fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
    f_t.savefig(path.join(results_dir, 't.png'), **savekws)
    f_lwe.savefig(path.join(results_dir, 'lwe.png'), **savekws)
    f_fr.savefig(path.join(results_dir, 'fr.png'), **savekws)
    f_n.savefig(path.join(results_dir, 'counts.png'), **savekws)
    f_azs.savefig(path.join(results_dir, 'azs.png'), **savekws)


