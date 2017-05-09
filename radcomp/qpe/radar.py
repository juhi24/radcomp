# coding: utf-8


class Radar:
    def __init__(self, lat=None, lon=None):
        self.lat = lat
        self.lon = lon


ker = Radar(lat=60.3881, lon=25.1139)
kum = Radar(lat=60.2045, lon=24.9633)
van = Radar(lat=60.2706, lon=24.8690)
RADARS = dict(KER=ker, KUM=kum, VAN=van)

