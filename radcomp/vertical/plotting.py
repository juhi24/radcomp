# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import radcomp.visualization as vis
import j24.visualization as jvis
from radcomp import vertical, learn
import pyart # for colormaps


DISPLACEMENT_FACTOR = 0.5
LABELS = dict(density='$\\rho$, kg$\,$m$^{-3}$',
              intensity='LWE, mm$\,$h$^{-1}$',
              liq='LWP, cm',
              FR='FR',
              temp_mean='$T$, $^{\circ}C$',
              azs='$\\alpha_{ZS}$')
DEFAULT_DISCRETE_CMAP = 'tab20'


def plot_data(data, ax=None, **kws):
    """plot Series"""
    if ax is None:
        ax = plt.gca()
    return ax.plot(data.index, data.values, drawstyle='steps', **kws)


def plot_vp(data, ax=None, **kws):
    """plot vertical profile"""
    if ax is None:
        ax = plt.gca()
    return ax.plot(data.values, data.index, **kws)


def plot_vps(df, axarr=None, fig_kws={'dpi': 110, 'figsize': (5, 3)}, **kws):
    """plot DataFrame of vertical profile parameters"""
    ncols = df.shape[1]
    if axarr is None:
        fig, axarr = plt.subplots(nrows=1, ncols=ncols, sharey=True, **fig_kws)
    else:
        fig = axarr[0].figure
    for i, (name, data) in enumerate(df.T.iterrows()):
        ax = axarr[i]
        plot_vp(data, ax=ax, **kws)
        search_name = name.upper()
        if search_name in vis.LABELS:
            ax.set_xlabel(vis.LABELS[search_name])
            ax.set_xlim(left=vis.VMINS[search_name], right=vis.VMAXS[search_name])
        else:
            ax.set_xlabel(name)
    set_h_ax(axarr[0])
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.15, top=0.9, wspace=0.1)
    for ax in axarr[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    return axarr


def rotate_tick_labels(rot, ax=None):
    if ax is None:
        ax = plt.gca()
    for tick in ax.get_xticklabels():
        tick.set_rotation(rot)


def mean_delta(t):
    dt = t[-1]-t[0]
    return dt/(len(t)-1)


def set_h_ax(ax, hlims=(0, 10000)):
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(vertical.m2km))
    ax.set_ylim(*hlims)
    ax.set_ylabel('Height, km')


def nice_cb_ticks(cb, nbins=5, steps=(1, 5, 10), **kws):
    # TODO: general for plotting, to be moved
    cb_tick_locator = mpl.ticker.MaxNLocator(nbins=nbins, steps=steps, **kws)
    cb.locator = cb_tick_locator
    cb.update_ticks()


def _pn_fig(fig_scale_factor, n_rows, **fig_kws):
    """Initialize figure for plotpn."""
    fw = fig_scale_factor*8
    fh = fig_scale_factor*(3+1.1*n_rows)
    return plt.figure(figsize=(fw,fh), **fig_kws)


def _pn_gs(fig_scale_factor, n_rows):
    """Initialize gridspec for plotpn."""
    left = 0.1*(11/(10+fig_scale_factor))
    right = 0.905*(10+fig_scale_factor)/11
    return mpl.gridspec.GridSpec(n_rows, 2, width_ratios=(35, 1), wspace=0.02,
                                 top=1-0.22/n_rows, bottom=0.35/n_rows,
                                 left=left, right=right)


def _pn_scalekws(field, scaled, x_is_date):
    """plotpn helper function for setting scaling arguments and cb label"""
    fieldup = field.upper()
    if scaled:
        scalekws = {'vmin': 0, 'vmax': 1}
        cb_label = 'scaled'
    elif fieldup in vis.LABELS:
        # custom limits
        vmins = vis.VMINS
        vmaxs = vis.VMAXS
        vmins['ZDR'] = -0.5
        if not x_is_date: # if not a time series
            vmaxs['ZDR'] = 2.5
        ##
        scalekws = {'vmin': vmins[fieldup],
                    'vmax': vmaxs[fieldup]}
        cb_label = vis.LABELS[fieldup]
    else:
        scalekws = {}
        cb_label = field
    return scalekws, cb_label


def _pn_x(df, x_is_date):
    """plotpn helper fucntion for setting x values"""
    if x_is_date:
        t = df.columns
        x = t - mean_delta(t)*DISPLACEMENT_FACTOR
    else:
        x = df.columns.sort_values()
    dx = mean_delta(x)
    x_last = x[-1:]+dx
    return x.append(x_last)


def plotpn(pn, fields=None, scaled=False, cmap='pyart_RefDiff', n_extra_ax=0,
           x_is_date=True, fig_scale_factor=0.65, fig_kws={'dpi': 150}, **kws):
    if fields is None:
        fields = pn.items
    n_rows = len(fields) + n_extra_ax
    fig = _pn_fig(fig_scale_factor, n_rows, **fig_kws)
    gs = _pn_gs(fig_scale_factor, n_rows)
    axarr = []
    for i, field in enumerate(fields):
        subplot_kws = {}
        if i > 0:
            subplot_kws['sharex'] = axarr[0]
        ax = fig.add_subplot(gs[i, 0], **subplot_kws)
        ax_cb = fig.add_subplot(gs[i, 1])
        axarr.append(ax)
        scalekws, cb_label = _pn_scalekws(field, scaled, x_is_date)
        kws.update(scalekws)
        x = _pn_x(pn[field], x_is_date)
        im = ax.pcolormesh(x, pn[field].index,
                           np.ma.masked_invalid(pn[field].values), cmap=cmap,
                           label=field, **kws)
        set_h_ax(ax)
        cb = fig.colorbar(im, cax=ax_cb, label=cb_label)
        nice_cb_ticks(cb)
    for j in range(n_extra_ax):
        ax = fig.add_subplot(gs[i+1+j, 0], sharex=axarr[0])
        axarr.append(ax)
    if x_is_date:
        axarr[-1].set_xlabel('Time, UTC')
        axarr[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        axarr[0].set_title(str(pn[field].columns[0].date()))
    # Hide xticks for all but last.
    for ax in axarr[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    return fig, axarr


def class_color(cid, cmap=DEFAULT_DISCRETE_CMAP, **kws):
    """j24.visualization.class_color wrapper"""
    jvis.class_color(cid, cmap=cmap, **kws)


def class_colors(classes, cmap=DEFAULT_DISCRETE_CMAP, **kws):
    """j24.visualization.class_colors wrapper"""
    jvis.class_colors(classes, cmap=cmap, **kws)


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
