# coding: utf-8
"""class statistics and comparison"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, plotting, RESULTS_DIR
from j24 import ensure_join
import conf


def consecutive_grouper(s):
    """Consecutive values to have same integer -> 111223333455"""
    return (s != s.shift()).cumsum()


def init_data(cases_id, scheme_id, has_ml=False, **kws):
    """initialize cases data"""
    cases = multicase.read_cases(cases_id)
    if has_ml:
        cases = cases[cases.ml_ok.astype(bool)]
    cc = multicase.MultiCase.by_combining(cases, has_ml=has_ml, **kws)
    cc.load_classification(scheme_id)
    for i, c in cases.case.iteritems():
        c.load_classification(scheme_id)
    return cases, cc


def init_snow(cases_id=conf.CASES_SNOW, scheme_id=conf.SCHEME_ID_SNOW):
    """initialize snow data"""
    return init_data(cases_id, scheme_id, has_ml=False)


def init_rain(cases_id=conf.CASES_MELT, scheme_id=conf.SCHEME_ID_MELT):
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
        g = c.classes.groupby(consecutive_grouper(c.classes))
        class_list.append(g.mean())
        count_list.append(g.count())
    classes = pd.concat(class_list)
    counts = pd.concat(count_list)
    counts.name = 'count'
    g_class_counts = counts.groupby(classes)
    return g_class_counts


def plot_class_streak_counts(cases):
    c = cases.case.iloc[0]
    fig, axarr, i = c.plot_cluster_centroids(plot_counts=False, n_extra_ax=1)
    if c.has_ml:
        axarr[0].set_ylabel('')
        axarr[1].set_ylabel('Height, km above ML top')
        axarr[2].set_ylabel('')
    ax = axarr[-1]
    class_streak_counts(cases).mean().plot.bar(ax=ax)
    plotting.bar_plot_colors(ax, i, class_color_fun=c.class_color, cm=plotting.cm_blue())
    ax.grid(axis='y')
    ax.set_ylabel('avg. occurrence\nstreak')
    ax.set_ylim(bottom=0, top=8)
    return fig, axarr


def occ_in_cases(cases, frac=True):
    """number or fraction of cases where each class occurs"""
    counts = []
    for cl_id in cases.case[0].class_scheme.get_class_list():
        count = 0
        for _, c in cases.case.iteritems():
            count += cl_id in c.classes.values
        counts.append(count)
    if frac:
        return np.array(counts)/cases.shape[0]
    return counts


def cl_frac_in_case(c, cl, frac=True):
    count = (c.classes==cl).sum()
    if frac:
        return count/c.classes.size
    return count


def frac_in_case_hist(cases, cl, log=True, ax=None, frac=True):
    """histogram of fraction of class occurrences in a case"""
    ax = ax or plt.gca()
    if frac:
        bins = (0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    else:
        bins = (1, 10, 20, 30, 40, 50, 60)
    frac_per_case = cases.case.apply(lambda x: cl_frac_in_case(x, cl, frac=frac))
    zeros_count = (frac_per_case<0.01).sum()
    ax.stem([0], [zeros_count], markerfmt='o', label='class not present')
    frac_per_case.hist(log=log, bins=bins, ax=ax, label='class occurrence')
    if frac:
        ax.set_xlim(-0.03, 0.9)
    ax.set_ylim(bottom=0.9, top=2e2)
    ax.set_title('Class {} occurrence histogram'.format(cl))
    word = 'Fraction' if frac else 'Number'
    ax.set_xlabel(word + ' of profiles per event')
    ax.set_ylabel('Number of events')
    ax.legend()
    return ax


if __name__ == '__main__':
    plt.close('all')
    cases_r, cc_r = init_rain()
    cases_s, cc_s = init_snow()
    savedir = conf.P1_FIG_DIR
    for cases in (cases_r, cases_s):
        fig, axarr = plot_class_streak_counts(cases)
        precip_type = 'rain' if cases.case.iloc[0].has_ml else 'snow'
        filename = 'centroids_streak_{}.png'.format(precip_type)
        savefile = path.join(savedir, filename)
        fig.savefig(savefile, bbox_inches='tight')
    fig_s, axarr_s, i = cc_s.plot_cluster_centroids(colorful_bars='blue')
    fig_r, axarr_r, i = cc_r.plot_cluster_centroids(colorful_bars='blue',
                                                    plot_conv_occ=True)
    axarr_r[0].set_ylabel('')
    axarr_r[1].set_ylabel('Height, km above ML top')
    axarr_r[2].set_ylabel('')
    fig_s.savefig(path.join(savedir, 'clusters_s.png'), bbox_inches='tight')
    fig_r.savefig(path.join(savedir, 'clusters_r.png'), bbox_inches='tight')
    fig_h, ax_h = plt.subplots(dpi=150, figsize=(4,3))
    frac_in_case_hist(cases, 16, log=True, frac=False, ax=ax_h)
    fig_h.savefig(path.join(savedir, 'occ_hist.png'))
