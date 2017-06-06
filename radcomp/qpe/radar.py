# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import matplotlib.pyplot as plt


class Radar:
    def __init__(self, lat=None, lon=None):
        self.lat = lat
        self.lon = lon

    def radar_pixel(self, lats, lons):
        """radar pixel coordinates from latitude and longitude grids"""
        latmid = int(lats.shape[0]/2)
        lonmid = int(lons.shape[1]/2)
        lat1 = lats[latmid, :]
        lon1 = lons[:, lonmid]
        y = np.abs(lat1-self.lat).argmin()
        x = np.abs(lon1-self.lon).argmin()
        return x, y

    def draw_marker(self, ax=None, marker='D', color='black', **kws):
        if ax is None:
            ax = plt.gca()
        return ax.plot(self.lon, self.lat, marker=marker, color=color, **kws)

KER = Radar(lat=60.3881, lon=25.1139)
KUM = Radar(lat=60.2045, lon=24.9633)
VAN = Radar(lat=60.2706, lon=24.8690)
RADARS = dict(KER=KER, KUM=KUM, VAN=VAN)

