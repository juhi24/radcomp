# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np
import matplotlib.pyplot as plt


class Radar:
    """
    A simple radar marker class.
    
    Attributes
    ----------
    lat : float
    lon : float
    name : str
    """

    def __init__(self, lat=None, lon=None, name=None):
        self.lat = lat
        self.lon = lon
        self.name = name

    def radar_pixel(self, lats, lons):
        """radar pixel coordinates from latitude and longitude grids"""
        latmid = int(lats.shape[0]/2)
        lonmid = int(lons.shape[1]/2)
        lat1 = lats[latmid, :]
        lon1 = lons[:, lonmid]
        y = np.abs(lat1-self.lat).argmin()
        x = np.abs(lon1-self.lon).argmin()
        return x, y

    def draw_marker(self, ax=None, marker='D', color='black', withname=False,
                    bbox=None, **kws):
        if ax is None:
            ax = plt.gca()
        marker = ax.plot(self.lon, self.lat, marker=marker, color=color, **kws)
        if withname:
            ax.text(self.lon+0.05, self.lat, self.name, bbox=bbox, **kws)
        return marker
        

KER = Radar(lat=60.3881, lon=25.1139, name='KER')
KUM = Radar(lat=60.2045, lon=24.9633, name='KUM')
VAN = Radar(lat=60.2706, lon=24.8690, name='VAN')
RADARS = dict(KER=KER, KUM=KUM, VAN=VAN)

