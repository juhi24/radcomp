# coding: utf-8

import numpy as np
from os import path
from rasterio.plot import show
import rasterio
import tempfile
from urllib.request import urlretrieve
from fmio import USER_DIR, basemap, fmi

if __name__ == '__main__':
    config_dir = USER_DIR
    keyfilepath = path.join(config_dir, 'api.key')
    radurl = fmi.gen_url(keyfilepath)
    with tempfile.TemporaryDirectory() as tmp_path:
        radarfilepath = path.join(tmp_path, 'Radar-suomi_dbz_eureffin.tif')
        urlretrieve(radurl, filename=radarfilepath)
        border = basemap.border()
        cities = basemap.cities()
        radar_data=rasterio.open(radarfilepath)
    radar_data
    dat = radar_data.read(1)
    mask = dat==65535
    d = dat.copy()*0.01
    d[mask] = 0
    datm = np.ma.MaskedArray(data=d, mask=d==0)
    nummask = np.ma.MaskedArray(data=dat, mask=~mask)
    ax = border.to_crs(radar_data.read_crs().data).plot(zorder=0, color='gray')
    if cities is not None:
        cities.to_crs(radar_data.read_crs().data).plot(zorder=5, color='black',
                                                       ax=ax, markersize=2)
    show(datm, transform=radar_data.transform, ax=ax, zorder=3)
    show(nummask, transform=radar_data.transform, ax=ax, zorder=4, alpha=.1,
         interpolation='bilinear')
    ax.axis('off')
    ax.set_xlim(left=-5.5e4)
    ax.set_ylim(top=7.8e6, bottom=6.42e6)


