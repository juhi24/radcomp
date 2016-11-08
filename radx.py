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

    def datetime(self, var='time'):
        times = self.data.variables[var]
        return nc.num2date(times[:], times.units)

    def distance(self, x0, y0, x1, y1):
        x = self.data.variables['x0']
        y = self.data.variables['y0']
        dx = x[x1]-x[x0]
        dy = y[y1]-y[y0]
        return np.sqrt(dx**2 + dy**2)

    def distance_from_radar(self, x, y):
        (x0, y0) = self.radar_pixel()
        return self.distance(x0, y0, x, y)

    def scantime(self):
        ts=self.datetime('time_bounds')[0]
        return ts[1]-ts[0]

    def task_name(self):
        """Try to guess scan task name."""
        dt = self.scantime().total_seconds()
        if self.site()=='KER':
            if dt > 120:
                return 'VOL_A'
            if dt < 50:
                return 'KER_FMIB'
        return ''

    def z_min(self, x, y):
        z_cal = -40.
        log = 2.
        dist_term = 20*np.log10(self.distance_from_radar(x, y))
        if self.task_name() == 'VOL_A':
            z_cal = -35.25
            log = 2.5
        return z_cal + log + dist_term
