# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, RESULTS_DIR
from j24 import ensure_dir

name = 'mlt_18eig17clus_pca'
interactive = True


def interpolate(mli):
    mli.loc[0] = 0
    mli.loc[0:310] = mli.loc[0:310].interpolate()


def fltr_median(lim):
    from scipy.ndimage.filters import median_filter
    na = lim.isnull()
    lim = pd.Series(index=lim.index, data=median_filter(lim.fillna(0), 7))
    lim[na] = np.nan


if __name__ == '__main__':
    plt.close('all')
    plt.ion() if interactive else plt.ioff()
    #cases = multicase.read_cases('mlt_test')
    cases = multicase.read_cases('melting')
    cases = cases[cases.ml_ok.isnull()]
    results_dir = ensure_dir(path.join(RESULTS_DIR, 'ml'))
    for caseid, row in cases.iterrows():
        c = row.case
        c.load_classification(name)
        fig, axarr = c.plot(params=['ZH', 'zdr', 'RHO', 'MLI'], cmap='viridis',
                            plot_fr=False, plot_t=False, plot_azs=False,
                            plot_snd=False, plot_silh=False, plot_classes=False)
        if not interactive:
            fname = path.join(results_dir, c.name()+'.png')
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
