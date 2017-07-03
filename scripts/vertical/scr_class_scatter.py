# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timedelta
from os import path
from radcomp.vertical import case, insitu, classification, plotting, RESULTS_DIR
from j24 import ensure_dir


plt.ion()
plt.close('all')
np.random.seed(0)
n_eigens = 25
n_clusters = 20
reduced = True
param = 'intensity'

scheme = classification.scheme_name(basename='14-16_t', n_eigens=n_eigens,
                                    n_clusters=n_clusters, reduced=reduced)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'class_scatter', scheme))

table = dict(density=insitu.TABLE_FILTERED_PKL, intensity=insitu.TABLE_PKL)

cm=mpl.cm.get_cmap('tab20')

cases = case.read_cases('14-16by_hand')
#g = pd.read_pickle(table[param])
classes_list = []
rate_list = []
#lwp_list = []
t_list = []
for name in cases.index:
    #data_g = g.loc[name]
    c = cases.case[name]
    c.load_classification(scheme)
    classes = c.classes
    classes.index = classes.index.round('1min')
    classes.name = 'class'
    classes_list.append(classes)
    #rate = c.time_weighted_mean(data_g[param])
    #rate.fillna(0, inplace=True)
    rate_list.append(c.lwe())
    #lwp_list.append(c.lwp())
    t_list.append(c.ground_temperature())
classes_all = pd.concat(classes_list)
rate_all = pd.concat(rate_list)
rate_all[rate_all==np.inf] = 0
#lwp_all = pd.concat(lwp_list)
t_all = pd.concat(t_list)
fig_rate = plotting.hists_by_class(rate_all, classes_all)
#fig_lwp = plotting.hists_by_class(lwp_all, classes_all)
fig_t = plotting.hists_by_class(t_all, classes_all)
fig_rate.savefig(path.join(results_dir, rate_all.name + '.png'))
#fig_lwp.savefig(path.join(results_dir, lwp_all.name + '.png'))
fig_t.savefig(path.join(results_dir, t_all.name + '.png'))
