# coding: utf-8
import numpy as np
#import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir


plt.ion()
#plt.close('all')
np.random.seed(0)
n_eigens = 25
n_clusters = 20
reduced = True
plot_by_class = False
scheme = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                    n_clusters=n_clusters, reduced=reduced)
#scheme = '2014rhi_{n}comp'.format(n=n_comp)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary', scheme))
save = True

cases = case.read_cases('training_baecc')
c = case.Case.by_combining(cases)
c.load_classification(scheme)

if plot_by_class:
    df = c.pcolor_classes(cmap='viridis')
f_cen, axarr_cen = c.plot_cluster_centroids()

if save:
    if plot_by_class:
        for i, fig in df.fig.iteritems():
            fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
    f_cen.savefig(path.join(results_dir, 'centroids.png'))


