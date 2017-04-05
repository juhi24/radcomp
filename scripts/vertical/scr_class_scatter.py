# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, insitu, RESULTS_DIR
from j24 import ensure_dir


plt.ion()
plt.close('all')
np.random.seed(0)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'class_scatter'))
n_comp = 20
param = 'intensity'

table = dict(density=insitu.TABLE_FILTERED_PKL, intensity=insitu.TABLE_PKL)
xmax = dict(density=600, intensity=4)
incr = dict(density=50, intensity=0.25)
xlabel = dict(density='$\\rho$, kg$\,$m$^{-3}$', intensity='LWE, mm$\,$h$^{-1}$')

cm=mpl.cm.get_cmap('Vega20')

cases = case.read_cases('analysis')
g = pd.read_pickle(table[param])
classes_list = []
rate_list = []
for name in cases.index:
    data_g = g.loc[name]
    c = cases.case[name]
    scheme = '2014rhi_{n}comp'.format(n=n_comp)
    c.load_classification(scheme)
    classes = c.classes
    classes.index = classes.index.round('1min')
    classes.name = 'class'
    classes_list.append(classes)
    base = c.base_minute()
    rate = insitu.time_weighted_mean(data_g, rule='15min', param=param,
                                     base=base)
    rate.fillna(0, inplace=True)
    rate_list.append(rate)
classes_all = pd.concat(classes_list)
rate_all = pd.concat(rate_list)
axarr = rate_all.hist(by=classes_all, sharex=True, sharey=True,
                      bins=np.arange(0, xmax[param], incr[param]))
axflat = axarr.flatten()
axflat[0].set_xlim(0, xmax[param])
fig = axflat[0].get_figure()
frameax = fig.add_subplot(111, frameon=False)
frameax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
frameax.set_xlabel(xlabel[param])
frameax.set_ylabel('count')
for i, ax in enumerate(axflat):
    iclass = int(float(ax.get_title()))
    ax.set_title('class {}'.format(iclass))
    for p in ax.patches:
        p.set_color(cm.colors[iclass])
plt.tight_layout()
fig.savefig(path.join(results_dir, param + '.png'))
