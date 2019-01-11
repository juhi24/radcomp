# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase

import conf


if __name__ == '__main__':
    #plt.close('all')
    cases = multicase.read_cases('jun12')
    c = cases.case.iloc[0]
    c.load_classification(conf.SCHEME_ID_MELT)
    #c.plot(params=['zdr', 'rho', 'zh', 'MLI'], interactive=False)
    c.plot(params=['kdp', 'kdpg', 'zdr', 'zdrg', 'zh'],
           plot_extras=['cl'], cmap='viridis',
           t_contour_ax_ind='all', t_levels=[-20, -10, -8, -3])
