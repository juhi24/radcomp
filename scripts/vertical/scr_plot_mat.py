# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case
from j24 import ensure_join


if __name__ == '__main__':
    plt.close('all')
    datadir = path.expanduser('~/DATA/IKA/20180818RHI')
    #datadir = path.expanduser('~/DATA/IKA/test')
    vpdir = path.expanduser('~/results/radcomp/vertical/vp_dmitri')
    vpfiles = []
    #vpfiles.append(path.join(vpdir, '20160818_IKA_VP_from_RHI.mat'))
    #vpfiles.append(path.join(vpdir, '20160818_IKA_VP_from_RHI_1km_median.mat'))
    #vpfiles.append(path.join(vpdir, '20160818_IKA_VP_from_RHI_1km_mean.mat'))
    vpfiles.append(path.join(vpdir, '20160818_IKA_VP_from_RHI_1km_median_maesaka.mat'))
    vpfiles.append(path.join(vpdir, '20160818_IKA_vprhi_1km_median_m.mat'))
    for vpfile in vpfiles:
        c = case.Case.from_mat(vpfile)
        figc, axarrc = c.plot(params=['ZH', 'KDP', 'kdp', 'ZDR', 'zdr', 'RHO'],
                              cmap='viridis', plot_snd=False)
        #figc, axarrc = c.plot(params=['KDP', 'kdp'], cmap='viridis', plot_snd=False)
        fname = path.basename(vpfile)
        axarrc[0].set_title(fname)
        figdir = ensure_join(vpdir, 'png')
        figpath = path.join(figdir, fname[:-3]+'png')
        figc.savefig(figpath, bbox_inches='tight')
