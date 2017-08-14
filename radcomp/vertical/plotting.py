# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from radcomp import vertical, learn, visualization

DISPLACEMENT_FACTOR = 0.5
LABELS = dict(density='$\\rho$, kg$\,$m$^{-3}$',
              intensity='LWE, mm$\,$h$^{-1}$',
              liq='LWP, cm',
              FR='rime mass fraction',
              temp_mean='Temperature, $^{\circ}C$')
DEFAULT_DISCRETE_CMAP = 'tab20'

def plot_data(data, ax, **kws):
    return ax.plot(data.index, data.values, drawstyle='steps', **kws)

def rotate_tick_labels(rot, ax=None):
    if ax is None:
        ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_rotation(rot)

def mean_delta(t):
    dt = t[-1]-t[0]
    return dt/(len(t)-1)

def plotpn(pn, fields=None, scaled=False, cmap='gist_ncar', n_extra_ax=0,
           x_is_date=True, **kws):
    if fields is None:
        fields = pn.items
    n_rows = len(fields) + n_extra_ax
    fig = plt.figure(figsize=(8,3+1.1*n_rows))
    gs = mpl.gridspec.GridSpec(n_rows, 2, width_ratios=(35, 1), wspace=0.02,
                               top=1-0.22/n_rows, bottom=0.35/n_rows, left=0.1,
                               right=0.905)
    axarr = []
    for i, field in enumerate(fields):
        subplot_kws = {}
        if i>0:
            subplot_kws['sharex'] = axarr[0]
        ax = fig.add_subplot(gs[i, 0], **subplot_kws)
        ax_cb = fig.add_subplot(gs[i, 1])
        axarr.append(ax)
        fieldup = field.upper()
        if scaled:
            scalekws = {'vmin': 0, 'vmax': 1}
            label = 'scaled'
        elif fieldup in visualization.LABELS:
            # custom limits
            vmins = visualization.VMINS
            vmaxs = visualization.VMAXS
            vmins['ZDR'] = -0.5
            if cmap=='gist_ncar':
                vmins['ZH'] = -15 # have 0 with nicer color
            if not x_is_date: # if not a time series
                vmaxs['ZDR'] = 2.5
            ##
            scalekws = {'vmin': vmins[fieldup],
                        'vmax': vmaxs[fieldup]}
            label = visualization.LABELS[fieldup]
        else:
            scalekws = {}
            label = field
        kws.update(scalekws)
        if x_is_date:
            t = pn[field].columns
            x = t - mean_delta(t)*DISPLACEMENT_FACTOR
        else:
            x = pn[field].columns.sort_values()
        dx = mean_delta(x)
        x_last = x[-1:]+dx
        x = x.append(x_last)
        im = ax.pcolormesh(x, pn[field].index,
                      np.ma.masked_invalid(pn[field].values), cmap=cmap,
                      label=field, **kws)
        #fig.autofmt_xdate()
        if x_is_date:
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(vertical.m2km))
        ax.set_ylim(0,10000)
        ax.set_ylabel('Height, km')
        #fig.colorbar(im, ax=ax, label=label)
        fig.colorbar(im, cax=ax_cb, label=label)
    for j in range(n_extra_ax):
        ax = fig.add_subplot(gs[i+1+j, 0], sharex=axarr[0])
        axarr.append(ax)
    if x_is_date:
        axarr[-1].set_xlabel('Time, UTC')
        axarr[0].set_title(str(pn[field].columns[0].date()))
    # Hide xticks for all but last.
    for ax in axarr[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    return fig, axarr

def class_color(cid, cm=None, mapping=None, default=(1, 1, 1)):
    """pick a color for cid using optional mapping"""
    if cm is None:
        cm = plt.get_cmap(DEFAULT_DISCRETE_CMAP)
    if mapping is not None:
        if cid in mapping.index:
            return cm.colors[mapping[cid]]
        return default
    return cm.colors[cid]

def class_colors(classes, ymin=-0.2, ymax=0, ax=None, alpha=1,
                 mapping=None, cmap=DEFAULT_DISCRETE_CMAP, **kws):
    """plot time series of color coding"""
    if isinstance(classes.index, pd.DatetimeIndex):
        t = classes.index
        dt = mean_delta(t)*DISPLACEMENT_FACTOR
        clss = classes.shift(freq=dt).dropna().astype(int)
    else:
        clss = classes
        dt = DISPLACEMENT_FACTOR
        clss.index = clss.index+dt
    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    t0 = clss.index[0]-2*dt
    for t1, cid in clss.iteritems():
        color = class_color(cid, cm, mapping=mapping)
        ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, facecolor=color,
                   alpha=alpha, clip_on=False, **kws)
        t0 = t1

def plot_classes(data, classes):
    figs = []
    axarrs = []
    n_classes = classes.max()+1
    for eigen in range(n_classes):
        i_classes = np.where(classes==eigen)[0]
        if len(i_classes)==0:
            continue
        pn_class = data.iloc[:, i_classes, :]
        fig, axarr = learn.plot_class(pn_class, ylim=(-1, 2))
        axarr[0].legend().set_visible(True)
        figs.append(fig)
        axarrs.append(axarr)
        for ax in axarr:
            if ax.xaxis.get_ticklabels()[0].get_visible():
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(vertical.m2km))
    return figs, axarrs

def pcolor_class(g, **kws):
    gt = g.transpose(0, 2, 1)
    gt.minor_axis = list(range(gt.shape[2]))
    fig, axarr = plotpn(gt, x_is_date=False, **kws)
    return axarr

def hists_by_class(data, classes, cmap=DEFAULT_DISCRETE_CMAP, **kws):
    """histograms of data grouping by class"""
    cm = plt.get_cmap(cmap)
    xmin = dict(density=0, intensity=0, liq=0, temp_mean=-15, FR=0)
    xmax = dict(density=500, intensity=4, liq=0.08, temp_mean=5, FR=1)
    incr = dict(density=50, intensity=0.25, liq=0.01, temp_mean=2, FR=0.1)
    param = data.name
    axarr = data.hist(by=classes, sharex=True, sharey=True, normed=True,
                      bins=np.arange(xmin[param], xmax[param], incr[param]))
    axflat = axarr.flatten()
    axflat[0].set_xlim(xmin[param], xmax[param])
    fig = axflat[0].get_figure()
    frameax = fig.add_subplot(111, frameon=False)
    frameax.tick_params(labelcolor='none', top='off', bottom='off', left='off',
                        right='off')
    frameax.set_xlabel(LABELS[param])
    frameax.set_ylabel('probability density')
    for i, ax in enumerate(axflat):
        rotate_tick_labels(0, ax=ax)
        try:
            iclass = int(float(ax.get_title()))
        except ValueError:
            continue
        titletext = '{}'.format(iclass)
        ax.set_title('')
        ax.text(0.88, 0.82, titletext, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        for p in ax.patches:
            p.set_color(class_color(iclass, cm, **kws))
    #plt.tight_layout()
    return fig
