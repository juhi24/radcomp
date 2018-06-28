# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import multicase, RESULTS_DIR
from j24 import ensure_join


case_set = 'erad18_samples'
name = 'mlt_18eig17clus_pca'
results_dir = ensure_join(RESULTS_DIR, 'classified', name, case_set)

if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases(case_set)
    for i, c in cases.case.iteritems():
        c.load_classification(name)
        fig, axarr = c.plot(n_extra_ax=0)#, cmap='viridis')
        fig.set_size_inches(2.5, 5)
        axarr[-1].set_xticks(axarr[-1].get_xticks()[1::2])
