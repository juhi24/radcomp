# coding: utf-8
"""class statistics and comparison"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, plotting, RESULTS_DIR
from j24 import ensure_join


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


def init_snow(cases_id='14-16by_hand', scheme_id='14-16_t08_zdr05_19eig19clus_pca'):
    """initialize snow data"""
    return init_data(cases_id, scheme_id, has_ml=False)


def init_rain(cases_id='melting', scheme_id='mlt_18eig17clus_pca'):
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


if __name__ == '__main__':
    plt.close('all')
    cases_r, cc_r = init_rain()
    cases_s, cc_s = init_snow()
    #
    for cases in (cases_r, cases_s):
        precip_type = 'rain' if cases.case.iloc[0].has_ml else 'snow'
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
        #
        fig, axarr, i = c.plot_cluster_centroids(plot_counts=False, n_extra_ax=1)
        ax = axarr[-1]
        g_class_counts.mean().plot.bar(ax=ax)
        plotting.bar_plot_colors(ax, i, class_color_fun=c.class_color, cm=plotting.cm_blue())
        ax.grid(axis='y')
        ax.set_ylabel('avg. occurrence\nstreak')
        ax.set_ylim(bottom=0, top=8)
        savedir = ensure_join(RESULTS_DIR, 'erad18')
        filename = 'centroids_streak_{}.png'.format(precip_type)
        savefile = path.join(savedir, filename)
        fig.savefig(savefile, bbox_inches='tight')
