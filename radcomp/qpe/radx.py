# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import netCDF4 as nc
import copy
import radcomp.visualization as vis
from radcomp.qpe.radar import RADARS

SITES = ['KUM', 'KER', 'VAN']


def equalize_ker_zmin(nc0, nc1):
    lvar0 = list(nc0.data.variables)
    lvar1 = list(nc1.data.variables)
    if 'UNKNOWN_ID_72' in lvar0 and 'UNKNOWN_ID_74' in lvar0:
        nc1.z_min = nc0.z_min
    if 'UNKNOWN_ID_72' in lvar1 and 'UNKNOWN_ID_74' in lvar1:
        nc0.z_min = nc1.z_min
    return nc0, nc1


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)


class RADXgrid:
    """RADX grid object"""
    def __init__(self, filepath, rwmode='r'):
        self.filepath = filepath
        self.rwmode = rwmode
        self.x = self.data.variables['x0'][:]
        self.y = self.data.variables['y0'][:]
        self._task_name = None
        self._radar_pixel = None
        self._z_min = None
        self.equalize_dbz = False

    @property
    def data(self):
        return nc.Dataset(self.filepath, self.rwmode)

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
        if 'kum-' in self.data.title:
            return 'KUM'
        if 'VANTAA' in self.data.title:
            return 'VAN'
        return '???'

    def radar(self):
        return RADARS[self.site()]

    def radar_latlon(self):
        """(deg_n, deg_e)"""
        return (self.radar().lat, self.radar().lon)

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

    def plot_r(self):
        return vis.plot_r(self.rainrate())

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

    def range_field(self):
        ran = self.data.variables['lat0'][:].copy()
        for (x,y), element in np.ndenumerate(ran):
            ran[y, x] = self.distance_from_radar(x, y)
        return ran

    def scantime(self):
        ts=self.datetime('time_bounds')[0]
        return ts[1]-ts[0]

    def elevationtime(self):
        """scan time per elevation"""
        return self.scantime()/self.data.dimensions['z0'].size

    def elevation_end_time(self, n=1):
        if n>self.data.dimensions['z0'].size:
            raise ValueError('There are not that many elevations.')
        return self.datetime('start_time')[0] + n*self.elevationtime()

    def z_min_xy(self, x, y):
        dist_term = 20*np.log10(self.distance_from_radar(x, y))
        if self.task_name == 'KER_FMIB' or (self.site()=='KER' and self.equalize_dbz):
            z_cal = -35.25
            log = 2.5
        elif self.task_name == 'VOL_A':
            z_cal = -46.62
            log = 1.5
        else:
            return
        return z_cal + log + dist_term

    @property
    def z_min(self):
        if self._z_min is None:
            z_min = copy.deepcopy(self.dbz_raw()[0,0,:,:])
            for (x, y), val in np.ndenumerate(z_min):
                z_min[x, y] = self.z_min_xy(x, y)
            self._z_min = z_min
        return self._z_min

    @z_min.setter
    def z_min(self, z_min):
        self._z_min = z_min

    def mask(self):
        if self.site() == 'KER':
            bad_dbz = self.dbz_raw()[0, 0, :, :].data < self.z_min.data
        else:
            bad_dbz = self.dbz_raw()[0, 0, :, :].mask
        low_rhohv = self.data.variables['RHOHV'][0, 0, :, :] < 0.85
        return np.logical_or(bad_dbz, low_rhohv)

