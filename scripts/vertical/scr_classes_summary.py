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

n_eigens = 25
n_clusters = 25
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

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

if plot_by_class:
    df = c.pcolor_classes(cmap='viridis')
f_cen, axarr_cen = c.plot_cluster_centroids(cmap='viridis', colorful_bars='blue')

if save:
    if plot_by_class:
        for i, fig in df.fig.iteritems():
            fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
    f_cen.savefig(path.join(results_dir, 'centroids.png'))


