# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, insitu, classification, RESULTS_DIR
from j24 import ensure_dir


plt.ion()
plt.close('all')
np.random.seed(0)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'class_scatter'))
n_eigens = 20
n_clusters = 20
reduced = True
param = 'intensity'

scheme = classification.scheme_name(basename='baecc_t', n_eigens=n_eigens,
                                    n_clusters=n_clusters, reduced=reduced)

table = dict(density=insitu.TABLE_FILTERED_PKL, intensity=insitu.TABLE_PKL)
XMAX = dict(density=600, intensity=2, liq=0.08)
incr = dict(density=50, intensity=0.25, liq=0.01)
XLABEL = dict(density='$\\rho$, kg$\,$m$^{-3}$',
              intensity='LWE, mm$\,$h$^{-1}$',
              liq='LWP, cm')

cm=mpl.cm.get_cmap('Vega20')

def hist(data, classes):
    param = data.name
    axarr = data.hist(by=classes, sharex=True, sharey=True,
                         bins=np.arange(0, XMAX[param], incr[param]))
    axflat = axarr.flatten()
    axflat[0].set_xlim(0, XMAX[param])
    fig = axflat[0].get_figure()
    frameax = fig.add_subplot(111, frameon=False)
    frameax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    frameax.set_xlabel(XLABEL[param])
    frameax.set_ylabel('count')
    for i, ax in enumerate(axflat):
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
        try:
            iclass = int(float(ax.get_title()))
        except ValueError:
            continue
        ax.set_title('class {}'.format(iclass))
        for p in ax.patches:
            p.set_color(cm.colors[iclass])
    #plt.tight_layout()
    fig.savefig(path.join(results_dir, param + '.png'))

cases = case.read_cases('training_baecc')
g = pd.read_pickle(table[param])
classes_list = []
rate_list = []
lwp_list = []
for name in cases.index:
    data_g = g.loc[name]
    c = cases.case[name]
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
    lwp_list.append(c.lwp())
classes_all = pd.concat(classes_list)
rate_all = pd.concat(rate_list)
lwp_all = pd.concat(lwp_list)
hist(rate_all, classes_all)
hist(lwp_all, classes_all)