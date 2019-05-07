# coding: utf-8
"""class statistics and comparison"""

import datetime
from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from radcomp.vertical import multicase, plotting, recase, RESULTS_DIR
from j24.datetools import strfdelta

import conf


def consecutive_grouper(s):
    """Consecutive values to have same integer -> 111223333455"""
    return (s != s.shift()).cumsum()


def init_data(cases_id, scheme_id, has_ml=False, **kws):
    """initialize cases data"""
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
    return cases, cc


def init_snow(cases_id=conf.CASES_SNOW, scheme_id=conf.SCHEME_ID_SNOW):
    """initialize snow data"""
    return init_data(cases_id, scheme_id, has_ml=False)


def init_rain(cases_id=conf.CASES_RAIN, scheme_id=conf.SCHEME_ID_RAIN):
    """initialize rain data"""
    return init_data(cases_id, scheme_id, has_ml=True)


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


def plot_occ_in_cases(cases, order, ax=None):
    ax = ax or plt.gca()
    c = cases.case.iloc[0]
    ax.bar(order, occ_in_cases(cases)*100, width=0.5)
    plotting.bar_plot_colors(ax, order, class_color_fun=c.vpc.class_color,
                             cm=plotting.cm_blue())
    ax.set_ylabel('% of events')
    ax.grid(axis='y')
    ax.set_ylim(bottom=0, top=100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels((0, '', 50, '', 100))
    return ax


def barplot_class_stats(fracs, class_color, ax=None):
    """bar plot wrapper"""
    ax = ax or plt.gca()
    fracs.plot.bar(ax=ax)
    plotting.bar_plot_colors(ax, fracs.index, class_color_fun=class_color,
                             cm=plotting.cm_blue())
    ax.grid(axis='y')
    return ax


def barplot_nanmean_class_frac(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_frac).median(axis=1), class_color, ax=ax)
    ax.set_ylabel('Median\nfraction')
    ax.set_ylim(bottom=0, top=0.3)


def barplot_mean_class_frac(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_frac).fillna(0).mean(axis=1),
                        class_color, ax=ax)
    ax.set_ylabel('mean occ.\nfraction')
    ax.set_ylim(bottom=0, top=0.8)


def barplot_nanmean_class_count(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_count).median(axis=1), class_color, ax=ax)
    ax.set_ylabel('Median\nprofile count')
    ax.set_ylim(bottom=0, top=30)


def barplot_mean_class_count(cases, class_color, ax=None):
    ax = ax or plt.gca()
    barplot_class_stats(class_agg(cases, agg_fun=class_count).fillna(0).mean(axis=1),
                        class_color, ax=ax)
    ax.set_ylabel('mean\ncount')
    ax.set_ylim(bottom=0, top=30)


if __name__ == '__main__':
    save = True
    plt.close('all')
    #cases_r, cc_r = init_rain()
    #cases_s, cc_s = init_snow()
    rain = dict(id='r', cases=cases_r, cc=cc_r, kws={'plot_conv_occ': -1},
                free_ax=1)
    snow = dict(id='s', cases=cases_s, cc=cc_s, kws={}, free_ax=2)
    savedir = conf.P1_FIG_DIR
    n_convective = sum([c.is_convective or False for i, c in cases_r.case.iteritems()])
    for d in (rain, snow):
        cases = d['cases']
        cc = d['cc']
        kws = d['kws']
        free_ax = d['free_ax']
        season = 'rain' if cc.has_ml else 'snow'
        class_color = cases.case[0].vpc.class_color
        kws.update(plot_counts=False, n_extra_ax=3, colorful_bars='blue', fig_kws={'dpi': 80})
        fig, axarr, i = cc.plot_cluster_centroids(fields=['zh'], fig_scale_factor=1.1, **kws)
        axarr[0].set_title('{} events'.format(season).capitalize())
        #plot_class_streak_counts(cases, ax=axarr[free_ax], order=i)
        plot_occ_in_cases(cases, order=i, ax=axarr[free_ax])
        #barplot_mean_class_frac(cases, class_color, ax=axarr[free_ax+2])
        barplot_nanmean_class_frac(cases, class_color, ax=axarr[free_ax+1])
        #barplot_mean_class_count(cases, class_color, ax=axarr[free_ax+4])
        barplot_nanmean_class_count(cases, class_color, ax=axarr[free_ax+2])
        fname = 'clusters_{}.png'.format(d['id'])
        if save:
            fig.savefig(path.join(savedir, fname), bbox_inches='tight')
    #fig_h, ax_h = plt.subplots(dpi=100, figsize=(4, 3))
    #frac_in_case_hist(cases, 5, log=False, frac=True, ax=ax_h)
    #if save:
    #    fig_h.savefig(path.join(savedir, 'occ_hist.png'))
