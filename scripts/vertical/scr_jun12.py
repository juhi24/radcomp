# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import case, multicase, plotting

import conf


if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases('jun12')
    c = cases.case.iloc[0]
    c.load_classification(conf.SCHEME_ID_MELT)
    #c.plot(params=['zdr', 'rho', 'zh', 'MLI'], interactive=False)
    c.load_model_data(variable='omega')
    fig, axarr = c.plot(params=['kdp', 'kdpg', 'zdr', 'zh', 'omega'],
                        plot_extras=['cl'], cmap='viridis',
                        t_contour_ax_ind='all', t_levels=[-20, -10, -8, -3],
                        n_extra_ax=3)
    zdr_dgz = case.proc_indicator(c.data, 'zdrg')
    kdp_dgz = case.proc_indicator(c.data, 'kdpg')
    kdp_hm = case.proc_indicator(c.data, 'kdpg', tlims=(-8, -3))
    ax_hm = axarr[-3]
    c.plot_series(kdp_hm, ax=ax_hm)
    #ax_hm.set_ylim((0, 0.25))
    ax_hm.set_yscale('log')
    ax_hm.set_ylabel('HM kdp')
    ax_kdp = ax=axarr[-2]
    c.plot_series(kdp_dgz, ax=ax_kdp)
    #ax_kdp.set_ylim((0, 0.5))
    ax_kdp.set_yscale('log')
    ax_kdp.set_ylabel('DGZ kdp')
    ax_zdr = axarr[-1]
    c.plot_series(zdr_dgz, ax=ax_zdr)
    #ax_zdr.set_ylim((0, 4))
    ax_zdr.set_yscale('log')
    ax_zdr.set_ylabel('DGZ zdr')
