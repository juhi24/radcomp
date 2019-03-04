# coding: utf-8
"""tools for working with collections of cases"""

from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from radcomp import USER_DIR
from radcomp.vertical import case, plotting


COL_START = 'start'
COL_END = 'end'


def read_case_times(name):
    """Read case starting and ending times from cases directory."""
    filepath = path.join(USER_DIR, 'cases', name + '.csv')
    dts = pd.read_csv(filepath, parse_dates=[COL_START, COL_END], comment='#',
                      skip_blank_lines=True)
    indexing_func = lambda row: case.case_id_fmt(row[COL_START], row[COL_END])
    dts.index = dts.apply(indexing_func, axis=1)
    dts.index.name = 'id'
    return dts


def _row_bool_flag(row, flag_name, default=None):
    """If exists, convert flag to boolean, else use default. None on empty."""
    if flag_name in row:
        if np.isnan(row[flag_name]):
            flag = None
        else:
            flag = bool(row[flag_name])
        return flag
    return default


def read_cases(name):
    """Read cases based on a cases list."""
    dts = read_case_times(name)
    cases_list = []
    for cid, row in dts.iterrows():
        case_kws = dict()
        case_kws['has_ml'] = _row_bool_flag(row, 'ml', default=False)
        case_kws['is_convective'] = _row_bool_flag(row, 'convective', default=None)
        try:
            t_start, t_end = row[COL_START], row[COL_END]
            c = case.Case.from_dtrange(t_start, t_end, **case_kws)
            if c.data.empty:
                err_msg_fmt = 'No data available between {} and {}.'
                raise ValueError(err_msg_fmt.format(t_start, t_end))
            cases_list.append(c)
        except ValueError as e:
            print('Error: {}. Skipping {}'.format(e, cid))
            dts.drop(cid, inplace=True)
    dts['case'] = cases_list
    return dts


def plot_convective_occurrence(occ, ax=None, **kws):
    """Bar plot convective occurrence.

    Args:
        occ (Series)
    """
    ax = ax or plt.gca()
    occ.plot.bar(ax=ax, **kws)
    ax.set_ylabel('norm. freq. in\nconvection')
    ax.yaxis.grid(True)


def ts_case_ids(cases):
    """case ids by timestamp"""
    cids_list = []
    for cid, c in cases.case.iteritems():
        cids_list.append(c.timestamps().apply(lambda x: cid))
    return pd.concat(cids_list)


def n_class_in_cases(class_n, cases, combined_cases=None):
    """numbers of occurrences of a given class per case in cases DataFrame"""
    case_ids = ts_case_ids(cases)
    if cases.case.iloc[0].vpc is not None:
        classes = pd.concat([c.vpc.classes for i, c in cases.case.iteritems()])
    elif combined_cases is not None:
        classes = combined_cases.classes
    groups = classes.groupby(case_ids)
    return groups.agg(lambda x: (x == class_n).sum())


def plot_cases_with_class(cases, class_n, **kws):
    selection = n_class_in_cases(class_n, cases).astype(bool)
    matching_cases = cases[selection]
    for name, c in matching_cases.case.iteritems():
        c.plot(**kws)


class MultiCase(case.Case):
    """A case object combined from multiple cases"""

    @classmethod
    def by_combining(cls, cases, has_ml=None, vpc=None, **kws):
        """Combine a DataFrame of Case objects into one."""
        for cid, c in cases.case.iteritems():
            c.load_model_temperature()
        has_ml = has_ml or c.has_ml
        vpc = vpc or c.vpc
        datas = list(cases.case.apply(lambda c: c.data)) # data of each case
        data = pd.concat(datas, axis=2)
        if 'convective' in cases:
            conv_flags = []
            for i, c in cases.case.iteritems():
                flag_series = c.timestamps().apply(lambda x: c.is_convective)
                conv_flags.append(flag_series)
            kws['is_convective'] = pd.concat(conv_flags)
        return cls(data=data, has_ml=has_ml, vpc=vpc, **kws)

    @classmethod
    def from_caselist(cls, name, filter_flag=None, **kws):
        """MultiCase from a case list using optional filter flag"""
        cases = read_cases(name)
        if filter_flag is not None:
            cases = cases[cases[filter_flag].fillna(0).astype(bool)]
        return cls.by_combining(cases, **kws)

    def class_convective_fraction(self):
        """fraction of convective profiles per class"""
        groups = self.is_convective.groupby(self.vpc.classes)
        return groups.agg(lambda x: x.sum()/x.count())

    def class_convective_rel_occ(self):
        """normalized frequenzy of convective profiles per class"""
        frac_total = self.is_convective.sum()/self.is_convective.count()
        return (self.class_convective_fraction()-frac_total)/frac_total + 1

    def plot_cluster_centroids(self, plot_conv_occ=False, n_extra_ax=0,
                               colorful_bars=False, **kws):
        """class centroids pcolormesh with optional extra stats"""
        n_extra_ax += plot_conv_occ
        kws.update(n_extra_ax=n_extra_ax, colorful_bars=colorful_bars)
        fig, axarr, order = self.vpc.plot_cluster_centroids(**kws)
        if plot_conv_occ:
            ax_conv = axarr[-2]
            occ = self.class_convective_rel_occ()
            plot_convective_occurrence(occ, ax=ax_conv)
            if colorful_bars:
                if colorful_bars == 'blue':
                    cmkw = {}
                    cmkw['cm'] = plotting.cm_blue()
                cmkw.update(class_color_fun=self.vpc.class_color)
                plotting.bar_plot_colors(ax_conv, order, **cmkw)
        return fig, axarr, order

    def t_surface(self, **kws):
        return super().t_surface(**kws).loc[self.data.minor_axis]
