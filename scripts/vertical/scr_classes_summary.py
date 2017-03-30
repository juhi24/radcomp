# coding: utf-8
import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, plotting, RESULTS_DIR
from j24 import ensure_dir, limitslist

SCALING_LIMITS = {'ZH': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3), 
                  'KDP': (0, 0.5), 'kdp': (0, 0.15)}

plt.ion()
plt.close('all')
np.random.seed(0)
n_comp = 20
scheme = '2014rhi_{n}comp'.format(n=n_comp)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary', scheme))

cases = case.read_cases('training')
c = case.Case.by_combining(cases)
c.load_classification(scheme)
#df = c.pcolor_classes()
#for i, fig in df.fig.iteritems():
#    fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))

def df2pn(df, limits):
    pass


clus_centers = pd.DataFrame(c.class_scheme.km.cluster_centers_.T)
lims = limitslist(np.arange(0,601,200))
dfs={}
for lim, param in zip(lims, c.class_scheme.params):
    df = clus_centers.iloc[lim[0]:lim[1],:]
    df.index = c.cl_data_scaled.minor_axis
    dfs[param] = df
pn = pd.Panel(dfs)
pn_decoded = case.scale_data(pn, reverse=True)
pn_plt = pn_decoded.copy()
pn_plt.minor_axis=pn_decoded.minor_axis-0.5
fig, axarr = plotting.plotpn(pn_plt, x_is_date=False)
ax=axarr[-1]
ax.set_xticks(range(n_comp))
ax.set_xlim(-0.5,n_comp-0.5)