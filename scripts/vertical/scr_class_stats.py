# coding: utf-8
"""class statistics and comparison"""

import datetime
import pickle
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from radcomp import CACHE_DIR
from radcomp.vertical import multicase, plotting, recase, RESULTS_DIR
from j24.datetools import strfdelta

import conf
import plot_wrappers


BOXPROPS = dict(whis=[2.5, 97.5], manage_xticks=False, sym='')


def consecutive_grouper(s):
    """Consecutive values to have same integer -> 111223333455"""
    return (s != s.shift()).cumsum()


def pkl_dump(var, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(var, f)


def pkl_load(filepath):
    """pickle.load wrapper"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def init_data(cases_id, scheme_id, has_ml=False, use_cache=False, **kws):
    """initialize cases data"""
    if use_cache:
        cases_fname = 'cases_{}{}.pkl'.format(cases_id, scheme_id)
        cases_file = path.join(CACHE_DIR, cases_fname)
        cc_fname = 'cc_{}{}.pkl'.format(cases_id, scheme_id)
        cc_file = path.join(CACHE_DIR, cc_fname)
        if path.exists(cases_file) and path.exists(cc_file):
            print('Using cached profile data.')
            return pkl_load(cases_file), pkl_load(cc_file)
    cases = multicase.read_cases(cases_id)
    if has_ml:
        cases = cases[cases.ml_ok.astype(bool)]
    for i, c in cases.case.iteritems():
        c.load_classification(scheme_id)
    cases, cc = recase.combine_cases_t_thresh(cases)
    #cc = multicase.MultiCase.by_combining(cases, has_ml=has_ml, **kws)
    cc.load_classification(scheme_id)
    for i, c in cases.case.iteritems():
        c.load_classification(scheme_id)
    if use_cache:
        pkl_dump(cases, cases_file)
        pkl_dump(cc, cc_file)
    return cases, cc


def init_snow(cases_id=conf.CASES_SNOW, scheme_id=conf.SCHEME_ID_SNOW, **kws):
    """initialize snow data"""
    return init_data(cases_id, scheme_id, has_ml=False, **kws)


def init_rain(cases_id=conf.CASES_RAIN, scheme_id=conf.SCHEME_ID_RAIN, **kws):
    """initialize rain data"""
    return init_data(cases_id, scheme_id, has_ml=True, **kws)


def streaks_table(g_counts):
    """df of class vs consecutive occurrence streak"""
    ocounts = {}
    for cl, g in g_counts:
        ocounts[cl] = g.groupby(g).count()
    count_df = pd.DataFrame(ocounts).fillna(0).astype(int)
    count_df.index.name = 'streak'
    return count_df


def class_streak_counts(cases):
    """average consecutive occurrence streak per class"""
    class_list = []
    count_list = []
    for i, c in cases.case.iteritems():
        g = c.classes().groupby(consecutive_grouper(c.classes()))
        class_list.append(g.mean())
        count_list.append(g.count())
    classes = pd.concat(class_list)
    counts = pd.concat(count_list)
    counts.name = 'count'
    g_class_counts = counts.groupby(classes)
    return g_class_counts


def class_streak_avg_time(cases):
    dt = cases.case.iloc[0].timedelta
    t = class_streak_counts(cases).mean()*dt-dt/2
    return t.apply(lambda x: x.total_seconds()/60)


def time_ticks(t, pos):
    dt = datetime.timedelta(minutes=t)
    return strfdelta(dt, '%H:%M')


def count2time_ticks(n, pos):
    return time_ticks(n*15, pos)


def class_count(classes):
    """count of each class in series of classes"""
    return classes.groupby(classes).count()


def class_frac(classes):
    """fraction of each class in series of classes"""
    return class_count(classes) / classes.shape[0]


def class_agg(cases, agg_fun=class_frac):
    """aggregate class occurrences"""
    fracs = [agg_fun(c.classes()) for i, c in cases.case.iteritems()]
    return pd.concat(fracs, axis=1)


def plot_class_streak_counts(cases, ax=None, order=None):
    c = cases.case.iloc[0]
    if ax is None:
        fig, axarr, order = c.plot_cluster_centroids(plot_counts=False, n_extra_ax=1)
        ax = axarr[-1]
    class_streak_avg_time(cases).plot.bar(ax=ax)
    plotting.bar_plot_colors(ax, order, class_color_fun=c.vpc.class_color, cm=plotting.cm_blue())
    ax.grid(axis='y')
    ax.set_ylabel('mean\npersistence')
    minorlocator = mpl.ticker.FixedLocator((20,30,40,50,2*60,3*60,4*60,5*60))
    #majorlocator = mpl.ticker.LogLocator(base=60)
    formatter = mpl.ticker.FuncFormatter(time_ticks)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim(bottom=15, top=7*60)
    ax.set_yscale('log')
    ax.yaxis.set_minor_locator(minorlocator)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    yticks = (10, 30, 60, 120, 5*60)
    ax.set_yticks(yticks)
    ax.set_yticklabels([time_ticks(tick, None) for tick in yticks])
    #ax.yaxis.set_major_locator(majorlocator)
    return ax


def occ_in_cases(cases, frac=True):
    """number or fraction of cases where each class occurs"""
    counts = []
    for cl_id in cases.case[0].vpc.get_class_list():
        count = 0
        for _, c in cases.case.iteritems():
            count += cl_id in c.classes().values
        counts.append(count)
    if frac:
        return np.array(counts)/cases.shape[0]
    return counts


def cl_frac_in_classes(classes, cl, frac=True):
    """fraction or number of class occurrences in a classes"""
    count = (classes==cl).sum()
    if frac:
        return count/classes.size
    return count


def frac_in_case_hist(cases, cl=-1, frac_per_case=None, log=True, ax=None, frac=True):
    """histogram of fraction of class occurrences in a case"""
    ax = ax or plt.gca()
    if frac:
        bins = np.concatenate(([0.001],np.arange(0.1, 1, 0.1)))
    else:
        bins = np.concatenate(([1],range(5, 65, 5)))
    if frac_per_case is None:
        agg_fun = lambda x: cl_frac_in_classes(x.classes(), cl, frac=frac)
        frac_per_case = cases.case.apply(agg_fun)
    zeros_count = (frac_per_case<0.01).sum()
    ax.stem([0], [zeros_count], markerfmt='o', label='class not present')
    frac_per_case.hist(log=log, bins=bins, ax=ax, label='class occurrence')
    if frac:
        ax.set_xlim(-0.03, 0.9)
    ax.set_ylim(bottom=0.9, top=1e2)
    ax.set_title('Class {} occurrence histogram'.format(cl))
    word = 'Fraction' if frac else 'Number'
    ax.set_xlabel(word + ' of profiles per event')
    ax.set_ylabel('Number of events')
    ax.legend()
    return ax


def plot_occ_in_cases(cases, order, class_color=None, ax=None):
    """class occurs in % of cases bar plot"""
    ax = ax or plt.gca()
    if class_color is None:
        color = plotting.cm_blue().colors[0]
        ax.bar(order, occ_in_cases(cases)*100, width=0.58, color=color)
    else:
        ax.bar(order, occ_in_cases(cases)*100, width=0.58)
        plotting.bar_plot_colors(ax, order, class_color_fun=class_color,
                                 cm=plotting.cm_blue())
    ax.set_ylabel('Frequency,\n% of events')
    ax.grid(axis='y')
    ax.set_ylim(bottom=0, top=100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels((0, '', 50, '', 100))
    return ax


def barplot_class_stats(fracs, class_color=None, ax=None):
    """bar plot wrapper"""
    ax = ax or plt.gca()
    if class_color is None:
        fracs.plot.bar(ax=ax, color=plotting.cm_blue().colors[0])
    else:
        fracs.plot.bar(ax=ax)
        plotting.bar_plot_colors(ax, fracs.index, class_color_fun=class_color,
                                 cm=plotting.cm_blue())
    ax.grid(axis='y')
    return ax


def barplot_nanmedian_class_frac(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_frac).median(axis=1), class_color, ax=ax)
    ax.set_ylabel('Median\nfraction')
    ax.set_ylim(bottom=0, top=0.3)


def boxplot_class_frac(cases, class_color, ax=None):
    ax = ax or plt.gca()
    pos = range(0, cases.case[0].vpc.n_clusters)
    class_agg(cases, agg_fun=class_frac).T.boxplot(ax=ax, positions=pos, **BOXPROPS)
    ax.set_ylabel('Fraction of profiles\nper event')
    ax.set_ylim(bottom=0, top=1.02)


def barplot_mean_class_frac(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_frac).fillna(0).mean(axis=1),
                        class_color, ax=ax)
    ax.set_ylabel('mean occ.\nfraction')
    ax.set_ylim(bottom=0, top=0.8)


def barplot_nanmedian_class_count(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_count).median(axis=1), class_color, ax=ax)
    ax.set_ylabel('Median\nprofile count')
    ax.set_ylim(bottom=0, top=30)


def boxplot_class_count(cases, class_color, ax=None):
    ax = ax or plt.gca()
    pos = range(0, cases.case[0].vpc.n_clusters)
    class_agg(cases, agg_fun=class_count).T.boxplot(ax=ax, positions=pos, **BOXPROPS)
    ax.set_ylabel('Profile count\nper event')
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.9, top=100)


def boxplot_class_time(cases, class_color, ax=None): # TODO
    ax = ax or plt.gca()
    pos = range(0, cases.case[0].vpc.n_clusters)
    locs = np.array([20,30,40,50,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,20*60])/15
    minorlocator = mpl.ticker.FixedLocator(locs)
    formatter = mpl.ticker.FuncFormatter(count2time_ticks)
    class_agg(cases, agg_fun=class_count).T.boxplot(ax=ax, positions=pos, **BOXPROPS)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_locator(minorlocator)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    yticks = np.array([10, 30, 60, 120, 5*60, 10*60])/15
    ax.set_yticks(yticks)
    ax.set_yticklabels([count2time_ticks(tick, None) for tick in yticks])
    ax.set_ylabel('Tot. duration\nper event')
    ax.set_ylim(bottom=10/15, top=16*60/15)


def barplot_mean_class_count(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_count).fillna(0).mean(axis=1),
                        class_color, ax=ax)
    ax.set_ylabel('mean\ncount')
    ax.set_ylim(bottom=0, top=30)


def axlabels(axarr, has_ml):
    """Label axes A1, A2, ..."""
    ab = 'a' if has_ml else 'b'
    x = 0.89 if has_ml else 0.94
    y = 0.89 if has_ml else 0.81
    for i, ax in enumerate(axarr):
        axlabel = '{}{}'.format(ab.capitalize(), i+1)
        color = 'white' if i==0 else 'black'
        ax.text(x, y, axlabel, verticalalignment='top', color=color,
                horizontalalignment='center', transform=ax.transAxes)


if __name__ == '__main__':
    plt.ion()
    save = True
    plt.close('all')
    use_cache = True
    cases_r, cc_r = init_rain(use_cache=use_cache)
    cases_s, cc_s = init_snow(use_cache=use_cache)
    rain = dict(id='R', cases=cases_r, cc=cc_r, kws={'plot_conv_occ': 1},
                free_ax=1)
    snow = dict(id='S', cases=cases_s, cc=cc_s, kws={'lim_override': True}, free_ax=1)
    savedir = conf.P1_FIG_DIR
    n_convective = sum([c.is_convective or False for i, c in cases_r.case.iteritems()])
    for d in (rain, snow):
        cases = d['cases']
        cc = d['cc']
        kws = d['kws']
        free_ax = d['free_ax']
        season = 'rain' if cc.has_ml else 'snow'
        class_color = cases.case[0].vpc.class_color
        fig_w_factor = 0.75 if cc.has_ml else 1.05
        fig_kws = dict(fig_h_factor=1.3, fig_w_factor=fig_w_factor)
        kws.update(plot_counts=False, n_extra_ax=2, colorful_bars='blue',
                   fig_scale_factor=0.68, fig_kws=fig_kws)
        fig, axarr, i = cc.plot_cluster_centroids(fields=['zh'], **kws)
        axarr[0].set_title('{season}-model'.format(season=season[0].capitalize()))
        #plot_class_streak_counts(cases, ax=axarr[free_ax], order=i)
        plot_occ_in_cases(cases, class_color=None, order=i, ax=axarr[free_ax+1])
        #barplot_mean_class_frac(cases, class_color, ax=axarr[free_ax+2])
        #boxplot_class_frac(cases, class_color, ax=axarr[free_ax+1])
        #barplot_mean_class_count(cases, class_color, ax=axarr[free_ax+4])
        boxplot_class_time(cases, class_color, ax=axarr[free_ax+2])
        plotting.set_h_label(axarr[0], cc.has_ml, narrow=True)
        plotting.prepend_class_xticks(axarr[-1], cc.has_ml)
        plotting.rotate_tick_labels(60, ax=axarr[-1])
        if not cc.has_ml:
            for i in (-1, -2):
                axarr[i].set_ylabel('')
        if not cc.has_ml:
            axarr[1].set_ylabel('$T_s$ at class\ncentroid, $^{\circ}$C')
        axlabels(axarr, cc.has_ml)
        fname = 'clusters_{}.png'.format(d['id'].lower())
        if save:
            fig.savefig(path.join(savedir, fname), bbox_inches='tight', dpi=300)
        #plot_wrappers.boxplot_t_combined(cc)
    #fig_h, ax_h = plt.subplots(dpi=100, figsize=(4, 3))
    #frac_in_case_hist(cases, 5, log=False, frac=True, ax=ax_h)
    #if save:
    #    fig_h.savefig(path.join(savedir, 'occ_hist.png'))
