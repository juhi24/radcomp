# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from os import path
from itertools import combinations

import pandas as pd

from radcomp import USER_DIR


BENCHMARK_DIR = path.join(USER_DIR, 'benchmark')
Q_DEFAULT = 'kdp_hm'


class VPCBenchmark:
    """score VPC classification results"""

    def __init__(self, man_analysis=None):
        self.man_analysis = man_analysis

    def fit(self, vpc):
        raise NotImplementedError


class ProcBenchmark(VPCBenchmark):
    """
    Compare VPC classification against a supervised process classification.
    """
    def __init__(self, **kws):
        super().__init__(**kws)
        self.data_fitted = None
        self.n_clusters = None

    @classmethod
    def from_csv(cls, name='fingerprint', fltr_q=None, **kws):
        """new instance from csv"""
        csv = path.join(BENCHMARK_DIR, name + '.csv')
        df = pd.read_csv(csv, parse_dates=['start', 'end'])
        dtypes = dict(ml=bool, kdp_hm=bool, kdp_dgz=bool, inv=bool)
        man_analysis = df.astype(dtypes)
        if fltr_q is not None:
            man_analysis.query(fltr_q, inplace=True)
        return cls(man_analysis=man_analysis, **kws)

    def fit(self, vpc):
        """Generate comparison with VPC."""
        dfs = []
        for start, row in self.man_analysis.iterrows():
            ser = vpc.training_result[row['start']:row['end']]
            df = pd.DataFrame(ser, columns=['cl'])
            for name, value in row.iloc[2:].iteritems():
                df[name] = value
            dfs.append(df)
        self.data_fitted = pd.concat(dfs)
        self.n_clusters = vpc.n_clusters

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

    def query_all(self, fun, procs={'kdp_hm', 'kdp_dgz'}):
        """Query process occurrences against all classes."""
        stats = []
        for proc in procs:
            # ignore rows with multiple process flags
            other = procs - set((proc,))
            q_not = ' & ~(' + ' | '.join(other) + ')' if len(other) > 0 else ''
            col = self.query_classes(fun, proc + q_not)
            col.name = proc
            stats.append(col)
        # query simultaneous process occurrences
        for pair in combinations(procs, 2):
            q = '{} & {}'.format(*pair)
            col = self.query_classes(fun, q)
            col.name = q.replace(' ', '')
            stats.append(col)
        return pd.concat(stats, axis=1)


if __name__ == '__main__':
    pb = ProcBenchmark.from_csv()
