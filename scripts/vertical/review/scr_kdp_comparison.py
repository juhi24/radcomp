# coding: utf-8

from os import path

import xarray as xr
import matplotlib.pyplot as plt

from radcomp.tools import rhi
from radcomp.vertical import case

SAVE_KWS = dict(bbox_inches='tight', dpi=300)
ika_path = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/IKA_final'


def gennc():
    dir_in = path.join(ika_path, 'tmp')
    out = path.join(ika_path, 'out')
    return rhi.nc_workflow(dir_in, out, kdp_debug=False,
                           recalculate_kdp=False)


if __name__ == '__main__':
    plt.close('all')
    savedir = path.expanduser('~/Asiakirjat/julkaisut/omat/vp_class/review/minor_revisions')
    ncname = '20140221_IKA_vprhi.nc'
    dirs = {'out_m': 'KDP Maesaka',
            #'out_s': 'KDP Schneebeli',
            #'out_v': 'KDP Vulpiani',
            'out_csu': 'KDP CSU'}
    cs = dict()
    for key, value in dirs.items():
        ncpath = path.join(ika_path, key, ncname)
        ds = xr.open_dataset(ncpath)
        c = case.Case.from_xarray(ds)
        cs[key] = c
    c = cs['out_m']
    c.data['kdp_csu'] = cs['out_csu'].data['kdp']
    fig, axarr = c.plot(params=['zh', 'kdp', 'kdp_csu'], plot_extras=[])
    kws = dict(x=0.02, y=0.7, loc='left')
    axarr[0].set_title('KDP Maesaka', **kws)
    axarr[1].set_title('KDP CSU', **kws)
    ax = c.data.to_frame().plot.scatter(x='kdp', y='kdp_csu', zorder=1)
    ax.plot([-1,1], [-1,1], color='black', zorder=-1)
    ax.set_ylim(-0.05, 0.25)
    ax.set_xlim(-0.05, 0.25)
    ax.set_aspect('equal', 'box')
    ax.get_figure().savefig(path.join(savedir, 'scatter.png'), **SAVE_KWS)
    fig.savefig(path.join(savedir, 'kdps.png'), **SAVE_KWS)


