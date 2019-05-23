# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
"""Radar plotting functions and common variables."""

VMINS = {'ZH': -10, 'ZDR': -0.25, 'RHO': 0.8, 'KDP': 0, 'DP': 0, 'PHIDP': 0,
         'R': 0.05, 'MLI': 0, 'ML': 0, 'RH': 0.9, 'LR': -0.5, 'KDPG': -8,
         'ZDRG': -0.1, 'OMEGA': -2, 'T': -30}
VMAXS = {'ZH': 35, 'ZDR': 2, 'RHO': 1, 'KDP': 0.25, 'DP': 360, 'PHIDP': 30,
         'R': 16, 'MLI': 10, 'ML': 1, 'RH': 1, 'LR': 0.5, 'KDPG': 8,
         'ZDRG': 0.1, 'OMEGA': 2, 'T': 5}
VMINS_RAIN = {}
VMAXS_RAIN = {'ZH': 40, 'KDP': 0.5, 'ZDR': 3}
VMINS_NARROW = {}
VMAXS_NARROW = {}
LABELS = {'ZH': '$Z_{e}$, dBZ',
          'ZDR': '$Z_{dr}$, dB',
          'RHO': '$\\rho_{hv}$',
          'KDP': '$K_{dp}$, $^{\circ}$ km$^{-1}$',
          'DP': 'deg',
          'PHIDP': 'deg',
          'R': 'rainrate, mm$\,$h$^{-1}$',
          'MLI': 'ML indicator',
          'ML': 'detected ML',
          'RH': 'RH',
          'LR': 'lapse rate',
          'KDPG': '$K_{dp}$ gradient',
          'ZDRG': '$Z_{dr}$ gradient',
          'OMEGA': '$\omega$, Pa/s',
          'T': '$T$, $^{\circ}$C'}


def plot_base(r, lon=None, lat=None, fig=None, ax=None, vmin=0.05, vmax=10,
               mask=None, cblabel='rain rate (mm/h)', cmap='jet',
               transform=None, **cbkws):
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
    if lon is not None and lat is not None and transform is not None:
        im = ax.pcolormesh(lon, lat, r_, transform=transform, **pc_kws)
        ax.set_ymargin(0)
        ax.set_xmargin(0)
    else:
        im = ax.pcolormesh(r_, **pc_kws)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, **cbkws)
    cb.set_label(cblabel)
    return fig, ax


def vlims(has_ml=False, narrow=False):
    """value limits for radar data visualization"""
    vmins, vmaxs = VMINS.copy(), VMAXS.copy()
    if has_ml:
        vmins.update(VMINS_RAIN)
        vmaxs.update(VMAXS_RAIN)
    return vmins, vmaxs


def _plotmeta(key):
    vmin = VMINS[key]
    vmax = VMAXS[key]
    label = LABELS[key]
    return dict(vmin=vmin, vmax=vmax, label=label)


def plot_r(r, **kws):
    meta = _plotmeta('R')
    mask = np.bitwise_or(np.isnan(r), r<0.05)
    return plot_base(r, vmin=meta['vmin'], vmax=meta['vmax'], mask=mask,
                     cblabel=meta['label'], cmap='jet', **kws)


def plot_kdp(kdp, **kws):
    meta = _plotmeta('KDP')
    mask = np.bitwise_or(kdp==128, np.isnan(kdp))
    return plot_base(kdp, vmin=0.1, vmax=0.5,
                     cblabel=meta['label'], mask=mask,
                     cmap='viridis', **kws)


def plot_dbz(dbz, **kws):
    meta = _plotmeta('ZH')
    mask = np.isnan(dbz)
    return plot_base(dbz, vmin=meta['vmin'], vmax=40, mask=mask,
                     cblabel=meta['label'], cmap='gist_ncar', **kws)
