#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc


def db2lin(db):
    return 10.**(db/10.)


def plot_rainmap(r):
    fig, ax = plt.subplots()
    r_ = r.copy()
    r_[r < 0.05] = np.nan
    plt.figure()
    cax = ax.imshow(r_, vmin=0.05, vmax=10)
    cb = fig.colorbar(cax)
    cb.set_label("rain rate (mm/h)")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


class RADXgrid:
    """RADX grid object"""
    def __init__(self, filepath, rwmode='r'):
        self.data = nc.Dataset(filepath, rwmode)

    def site(self):
        if 'Kerava' in self.data.title:
            return 'KER'
        if 'kum' in self.data.title:
            return 'KUM'
        if 'VANTAA' in self.data.title:
            return 'VAN'
        return '???'

    def radar_latlon(self):
        """(deg_n, deg_e)"""
        if self.site() == 'KER':
            return (60+23.3/60, 25+6.8/60)
        return (np.nan, np.nan)

    def radar_pixel(self):
        """(idy, idx)"""
        ymid = self.data.dimensions['y0'].size/2
        xmid = self.data.dimensions['x0'].size/2
        lats = self.data.variables['lat0'][:,ymid]
        lons = self.data.variables['lon0'][xmid,:]
        lat, lon = self.radar_latlon()
        y = np.abs(lats-lat).argmin()
        x = np.abs(lons-lon).argmin()
        return (x, y)

    def rainrate(self):
        dbzdata = self.dbz()[0,0,:,:]
        if self.site() == 'KER':
            dbzdata_corrected = dbzdata+17
        else:
            dbzdata_corrected = dbzdata+2
        z = db2lin(dbzdata_corrected)
        return 0.0292*z**(0.6536)

    def dbz(self):
        if self.site() == 'KER':
            return self.data.variables['DBZ_TOT']
        return self.data.variables['DBZ']

    def plot_rainmap(self):
        return plot_rainmap(self.rainrate())

    def datetime(self):
        times = self.data.variables['time']
        return nc.num2date(times[:], times.units)