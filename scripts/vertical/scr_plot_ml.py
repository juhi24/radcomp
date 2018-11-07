# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import multicase, RESULTS_DIR
from j24 import ensure_dir
from conf import SCHEME_ID_MELT


name = SCHEME_ID_MELT
#name = 'mlt2_3eig10clus_pca'
interactive = True


if __name__ == '__main__':
    plt.close('all')
    plt.ion() if interactive else plt.ioff()
    cases = multicase.read_cases('mlt_test')
    #cases = multicase.read_cases('melting')
    #cases = cases[cases.ml_ok.isnull()]
    #cases = cases[~cases.ml_ok.astype(bool)]
    results_dir = ensure_dir(path.join(RESULTS_DIR, 'ml2'))
    for caseid, row in cases.iterrows():
        c = row.case
        print(c.name())
        c.load_classification(name)
        fig, axarr = c.plot(params=['zh', 'zdr', 'RHO', 'MLI'], cmap='viridis',
                            plot_fr=False, plot_t=False, plot_azs=False,
                            plot_silh=True, plot_classes=False,
                            t_contour_ax_ind=[0], t_levels=[0])
        if not interactive:
            fname = path.join(results_dir, c.name()+'.png')
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
