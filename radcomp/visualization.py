# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

VMINS = {'ZH': -10, 'ZDR': -1, 'RHO': 0, 'KDP': 0, 'DP': 0, 'PHIDP': 0,
         'R': 0.05}
VMINS_CB = dict(VMINS) # copy
VMINS_CB['ZH'] = -15
VMAXS = {'ZH': 30, 'ZDR': 4, 'RHO': 1, 'KDP': 0.26, 'DP': 360, 'PHIDP': 360,
         'R': 16}
LABELS = {'ZH': '$Z$, dBZ', 'ZDR': '$Z_{dr}$, dB', 'KDP': '$K_{dp}$, deg/km',
          'DP': 'deg', 'PHIDP': 'deg', 'R': 'rainrate, mm$\,$h$^{-1}$'}


def _plot_base(r, lon=None, lat=None, fig=None, ax=None, vmin=0.05, vmax=10, mask=None,
              cblabel='rain rate (mm/h)', cmap='jet',
              transform=ccrs.Orthographic(central_longitude=25, central_latitude=60),
              **cbkws):
    """base function for pcolormesh plotting"""
    if ax is None:
        fig, ax = plt.subplots()
    if fig is None:
        fig = ax.figure
    r_ = r.copy()
    if mask is not None:
        r_ = np.ma.masked_where(mask, r_)
    else:
        r_ = np.ma.masked_where(np.isnan(r), r_)
    pc_kws = dict(vmin=vmin, vmax=vmax, cmap=cmap)
    if lon is not None and lat is not None:
        im = ax.pcolormesh(lon, lat, r_, transform=transform, **pc_kws)
    else:
        im = ax.pcolormesh(r_, **pc_kws)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, **cbkws)
    cb.set_label(cblabel)
    return fig, ax


def _plotmeta(key):
    vmin = VMINS[key]
    vmax = VMAXS[key]
    label = LABELS[key]
    return dict(vmin=vmin, vmax=vmax, label=label)


def plot_r(r, **kws):
    meta = _plotmeta('R')
    mask = np.bitwise_or(np.isnan(r), r<0.05)
    return _plot_base(r, vmin=meta['vmin'], vmax=meta['vmax'], mask=mask,
                     cblabel=meta['label'], cmap='jet', **kws)


def plot_kdp(kdp, **kws):
    meta = _plotmeta('KDP')
    mask = np.bitwise_or(kdp==128, np.isnan(kdp))
    return _plot_base(kdp, vmin=0.1, vmax=0.5,
                     cblabel=meta['label'], mask=mask,
                     cmap='viridis', **kws)


def plot_dbz(dbz, **kws):
    meta = _plotmeta('ZH')
    mask = np.isnan(dbz)
    return _plot_base(dbz, vmin=meta['vmin'], vmax=40, mask=mask,
                     cblabel=meta['label'], cmap='gist_ncar', **kws)
