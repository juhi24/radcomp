#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Jussi Tiira
"""
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import copy


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
        self.x = self.data.variables['x0'][:]
        self.y = self.data.variables['y0'][:]
        self._task_name = None
        self._radar_pixel = None

    @property
    def task_name(self):
        """Try to guess scan task name."""
        if self._task_name is None:
            dt = self.scantime().total_seconds()
            if self.site()=='KER':
                if dt > 120:
                    self._task_name = 'VOL_A'
                if dt < 50:
                    self._task_name = 'KER_FMIB'
        return self._task_name

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

    @property
    def radar_pixel(self):
        """(idy, idx)"""
        if self._radar_pixel is None:
            ymid = self.data.dimensions['y0'].size/2
            xmid = self.data.dimensions['x0'].size/2
            lats = self.data.variables['lat0'][:,ymid]
            lons = self.data.variables['lon0'][xmid,:]
            lat, lon = self.radar_latlon()
            y = np.abs(lats-lat).argmin()
            x = np.abs(lons-lon).argmin()
            self._radar_pixel = (x, y)
        return self._radar_pixel

    def z(self):
        return db2lin(self.dbz())

    def rainrate(self):
        return 0.0292*self.z()**(0.6536)

    def dbz_raw(self):
        if self.site() == 'KER':
            return self.data.variables['DBZ_TOT']
        return self.data.variables['DBZ']

    def dbz(self):
        """filtered and corrected DBZ"""
        dbz = self.dbz_raw()[0,0,:,:]
        if self.site() == 'KER':
            dbz_corrected = dbz+17
        else:
            dbz_corrected = dbz+2
        dbz_corrected.mask = self.mask()
        return dbz_corrected

    def plot_rainmap(self):
        return plot_rainmap(self.rainrate())

    def datetime(self, var='time'):
        times = self.data.variables[var]
        return nc.num2date(times[:], times.units)

    def distance(self, x0, y0, x1, y1):
        dx = self.x[x1]-self.x[x0]
        dy = self.y[y1]-self.y[y0]
        return np.sqrt(dx**2 + dy**2)

    def distance_from_radar(self, x, y):
        (x0, y0) = self.radar_pixel
        return self.distance(x0, y0, x, y)

    def scantime(self):
        ts=self.datetime('time_bounds')[0]
        return ts[1]-ts[0]

    def z_min_xy(self, x, y):
        dist_term = 20*np.log10(self.distance_from_radar(x, y))
        if self.task_name == 'KER_FMIB':
            z_cal = -35.25
            log = 2.5
        elif self.task_name == 'VOL_A':
            z_cal = -46.62
            log = 1.5
        else:
            return
        return z_cal + log + dist_term

    def z_min(self):
        z_min = copy.deepcopy(self.dbz_raw()[0,0,:,:])
        for (x, y), val in np.ndenumerate(z_min):
            z_min[x, y] = self.z_min_xy(x, y)
        return z_min

    def mask(self):
        return self.dbz_raw()[0, 0, :, :].data < self.z_min().data

