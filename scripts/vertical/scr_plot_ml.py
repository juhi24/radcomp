# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, RESULTS_DIR
from j24 import ensure_dir

from scr_dynbottom import scheme


if __name__ == '__main__':
    plt.close('all')
    plt.ioff()
    cases = case.read_cases('melting')
    for caseid, row in cases.iterrows():
        c = row.case
        c.class_scheme = scheme
        fig, axarr = c.plot(params=['ZH', 'kdp', 'zdr', 'RHO'], cmap='viridis',
                            ml_iax=3)
        results_dir = ensure_dir(path.join(RESULTS_DIR, 'ml'))
        fname = path.join(results_dir, c.name()+'.png')
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
