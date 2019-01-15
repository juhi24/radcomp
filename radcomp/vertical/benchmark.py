# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path
from itertools import combinations

import numpy as np
import pandas as pd

from radcomp import USER_DIR


BENCHMARK_DIR = path.join(USER_DIR, 'benchmark')
Q_DEFAULT = 'kdp_hm'


def _data_by_bm(bm, c):
    return c.data.loc[:, :, bm.data_fitted.index]


def autoref(pn, rain_season=False):
    """automatic reference generation using case data"""
    from radcomp.vertical import case
    zdr_dgz = case.proc_indicator(pn, 'zdrg')
    kdp_dgz = case.proc_indicator(pn, 'kdpg')
    kdp_hm = case.proc_indicator(pn, 'kdpg', tlims=(-8, -3))
    df = pd.DataFrame(index=zdr_dgz.index)
    kdpmax = pn['kdp'].max()
    df['dgz_zdr'] = zdr_dgz > 0.15
    kdp_dgz_thresh = 0.15 if rain_season else 0.12
    kdp_hm_thresh = 0.1 if rain_season else 0.08
    df['dgz_kdp'] = (kdp_dgz > 0.03) & (kdpmax > kdp_dgz_thresh)
    df['hm_kdp'] = (kdp_hm > 0.02) & (kdpmax > kdp_hm_thresh)
    return df


class VPCBenchmark:
    """score VPC classification results"""

    def __init__(self, data=None):
        self.data = data
        self.data_fitted = None
        self.n_clusters = None

    def fit(self, vpc):
        raise NotImplementedError

    def query_count(self, cl, q=Q_DEFAULT):
        """number of matched query for a given class"""
        df = self.data_fitted
        query = 'cl==@cl & {}'.format(q)
        return df.query(query).shape[0]

    def query_frac(self, cl, q=Q_DEFAULT):
        """fraction of matched query for a given class"""
        df = self.data_fitted
        n = self.query_count(cl, q=q)
        n_cl_occ = (df['cl'] == cl).sum()
        return n/n_cl_occ

    def query_classes(self, fun, q=Q_DEFAULT):
        """fractions of matched query per class"""
        stat = pd.Series(index=range(self.n_clusters),
                         data=range(self.n_clusters))
        return stat.apply(fun, q=q)

    def query_all(self, fun, procs={'hm_kdp', 'dgz_kdp'}):
        """Query process occurrences against all classes."""
        stats = []
        # single process occurrences
        for proc in procs:
            # ignore rows with multiple process flags
            other = procs - set((proc,))
            q_not = ' & ~(' + ' | '.join(other) + ')' if len(other) > 0 else ''
            col = self.query_classes(fun, proc + q_not)
            col.name = proc
            stats.append(col)
        # query simultaneous process occurrences
        if len(procs)>1:
            for n_comb in np.arange(2, len(proc)+1):
                for comb in combinations(procs, n_comb):
                    q = ' & '.join(['{}' for i in range(n_comb)]).format(*comb)
                    other = procs - set(comb)
                    q_not = ' & ~(' + ' | '.join(other) + ')' if len(other) > 0 else ''
                    col = self.query_classes(fun, q + q_not)
                    col.name = q.replace(' ', '')
                    stats.append(col)
        # non-events
        q = ' & '.join(['~{}' for i in range(len(procs))]).format(*procs)
        col = self.query_classes(fun, q)
        col.name = 'non-event'
        stats.append(col)
        return pd.concat(stats, axis=1)


class AutoBenchmark(VPCBenchmark):
    """Compare VPC classification against other process classification."""

    def fit(self, vpc):
        cl = vpc.training_result.loc[self.data.index]
        df = self.data.copy()
        df['cl'] = cl
        self.data_fitted = df
        self.n_clusters = vpc.n_clusters


class ManBenchmark(VPCBenchmark):
    """
    Compare VPC classification against a supervised process classification.
    """

    @classmethod
    def from_csv(cls, name='fingerprint', fltr_q=None, **kws):
        """new instance from csv"""
        csv = path.join(BENCHMARK_DIR, name + '.csv')
        df = pd.read_csv(csv, parse_dates=['start', 'end'])
        dtypes = dict(ml=bool, hm_kdp=bool, dgz_kdp=bool, inv=bool)
        data = df.astype(dtypes)
        if fltr_q is not None:
            data.query(fltr_q, inplace=True)
        return cls(data=data, **kws)

    def fit(self, vpc):
        """Generate comparison with VPC."""
        dfs = []
        for start, row in self.data.iterrows():
            ser = vpc.training_result[row['start']:row['end']]
            df = pd.DataFrame(ser, columns=['cl'])
            for name, value in row.iloc[2:].iteritems():
                df[name] = value
            dfs.append(df)
        self.data_fitted = pd.concat(dfs)
        self.n_clusters = vpc.n_clusters

