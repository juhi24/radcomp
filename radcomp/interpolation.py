# coding: utf-8
import itertools
import datetime
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from os import path
from pyoptflow import utils
from pyoptflow.core import extract_motion_proesmans
from pyoptflow.interpolation import interpolate
from radcomp import radx
from j24 import ensure_dir


def interp(I1, I2, n=1):
    """Interpolate n frames."""
    VF,VB = motion(I1, I2)
    # Interpolate n frames between the input images.
    return interpolate(I1, I2, VF, n, VB=VB)


def motion(I1, I2):
    # Convert the rainfall maps to unsigned byte, as required by the Optflow 
    # motion detection algorithms. Gaussian filter with std. dev. 3 is applied.
    Iu = []
    for i, I in enumerate([I1, I2]):
        Iu.append(utils.rainfall_to_ubyte(I, R_min=0.05, R_max=10.0, filter_stddev=3.0))
    # Compute the motion field by using the Python <-> C++ API and the Proesmans 
    # algorithm.
    return extract_motion_proesmans(Iu[0], Iu[1], lam=25.0, num_iter=250, num_levels=6)


def batch_interpolate(filepaths_good, outpath, data_site=None, save_png=False,
                      interval_s=10.):
    #elev = []
    #l_dt = []
    #l_t0_str = []
    #l_t1_str = []
    interval_dt = datetime.timedelta(seconds=interval_s)
    for f0, f1 in itertools.izip(filepaths_good, filepaths_good[1:]):
        if data_site is None:
            for site in radx.SITES:
                if site in f0:
                    data_site = site
        nc0 = radx.RADXgrid(f0)
        nc1 = radx.RADXgrid(f1)
        if data_site == 'KER':
            nc0, nc1 = radx.equalize_ker_zmin(nc0, nc1)
        #elev.append(nc0.variables['z0'][0])
        I1 = nc0.rainrate()
        I2 = nc1.rainrate()
        t0 = nc0.elevation_end_time()
        t1 = nc1.elevation_end_time()
        dt = t1-t0
        #l_t0_str.append(str(t0))
        #l_t1_str.append(str(t1))
        n = int(round(dt.total_seconds()/interval_s))
        intrp_timestamps = [t0+i*interval_dt for i in range(1, n+1, 1)]
        selection = [t.second<10 for t in intrp_timestamps]
        #l_dt.append(dt)
        print(dt.total_seconds())
        intrp = np.array(interp(I1, I2, n))
        for i in np.where(selection)[0]:
            t = intrp_timestamps[i]
            r = intrp[i]
            datedir = t.strftime('%Y%m%d')
            fbasename = t.strftime('intrp_%Y%m%d_%H%M%S')
            matfname = fbasename + '.mat'
            matsitepath = ensure_dir(path.join(outpath, data_site, 'R', 'mat', datedir))
            matfilepath = path.join(matsitepath, matfname)
            mdict = {'time': np.array(str(t)), 'R': r}
            scipy.io.savemat(matfilepath, mdict, do_compression=True)
            if save_png:
                pngfname = fbasename + '.png'
                pngsitepath = ensure_dir(path.join(outpath, data_site, 'R', 'png', datedir))
                pngfilepath = path.join(pngsitepath, pngfname)
                fig, ax = radx.plot_rainmap(r)
                ax.set_title(str(t))
                fig.savefig(pngfilepath, bbox_inches="tight")
                plt.close(fig)
