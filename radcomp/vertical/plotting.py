# coding: utf-8
#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from radcomp import vertical, learn

VMINS = {'ZH': -10, 'ZDR': -1, 'RHO': 0, 'KDP': 0, 'DP': 0, 'PHIDP': 0}
VMINS_CB = dict(VMINS)
VMINS_CB['ZH'] = -15
VMAXS = {'ZH': 30, 'ZDR': 4, 'RHO': 1, 'KDP': 0.26, 'DP': 360, 'PHIDP': 360}
LABELS = {'ZH': '$Z$, dBZ', 'ZDR': '$Z_{dr}$, dB', 'KDP': '$K_{dp}$, deg/km',
          'DP': 'deg', 'PHIDP': 'deg'}
DISPLACEMENT_FACTOR = 0.5

def plot_data(data, ax, **kws):
    return ax.plot(data.index, data.values, drawstyle='steps', **kws)

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
                           top=1-0.22/n_rows, bottom=0.35/n_rows, left=0.1, right=0.905)
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
        elif fieldup in LABELS:
            scalekws = {'vmin': VMINS_CB[fieldup], 'vmax': VMAXS[fieldup]}
            label = LABELS[fieldup]
        else:
            scalekws = {}
            label = field
        if x_is_date:
            t = pn[field].columns
            x = t - mean_delta(t)*DISPLACEMENT_FACTOR
            x
        else:
            x = pn[field].columns
        dx = mean_delta(x)
        x_last = x[-1:]+dx
        x = x.append(x_last)
        im = ax.pcolormesh(x, pn[field].index, 
                      np.ma.masked_invalid(pn[field].values), cmap=cmap,
                      **scalekws, label=field, **kws)
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
    for ax in axarr[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    return fig, axarr

def class_colors(classes, ymin=-0.2, ymax=0, ax=None, cmap='Vega20', alpha=1, **kws):
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
    for t1, icolor in clss.iteritems():
        if t1<=t0:
            print('HALP')
        ax.axvspan(t0, t1, ymin=ymin, ymax=ymax, facecolor=cm.colors[icolor], 
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

def hists_by_class(data, classes):
    """histograms of data grouping by class"""
    cm = mpl.cm.get_cmap('Vega20')
    xmin = dict(density=0, intensity=0, liq=0, temp_mean=-15)
    xmax = dict(density=500, intensity=2, liq=0.08, temp_mean=5)
    incr = dict(density=50, intensity=0.25, liq=0.01, temp_mean=2)
    xlabel = dict(density='$\\rho$, kg$\,$m$^{-3}$',
                  intensity='LWE, mm$\,$h$^{-1}$',
                  liq='LWP, cm',
                  temp_mean='Temperature, $^{\circ}C$')
    param = data.name
    axarr = data.hist(by=classes, sharex=True, sharey=True,
                         bins=np.arange(xmin[param], xmax[param], incr[param]))
    axflat = axarr.flatten()
    axflat[0].set_xlim(xmin[param], xmax[param])
    fig = axflat[0].get_figure()
    frameax = fig.add_subplot(111, frameon=False)
    frameax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    frameax.set_xlabel(xlabel[param])
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
    return fig
