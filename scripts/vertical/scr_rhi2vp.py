# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Adapted from a script by Dmitri Moisseev.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pyart
import numpy as np
import copy
import glob
import scipy.io as sio
from os import path
from datetime import datetime


def main(pathIn, pathOut):
    """Extract profiles and save as mat."""
    file_indx = 0
    files = np.sort(glob.glob(path.join(pathIn, "*RHI_HV*.raw")))
    nfile = len(files)

    zh_vp    = np.zeros([nfile, 300])
    zdr_vp   = np.zeros([nfile, 300])
    kdp_vp   = np.zeros([nfile, 300])
    rho_vp   = np.zeros([nfile, 300])
    dp_vp    = np.zeros([nfile, 300])
    ObsTime  = []
    height   = np.linspace(0,15000,300)
    for filename in files:
        try: ## reading radar data
            radar = pyart.io.read(filename)

            ### Fixing Zdr calibration
            zdr= copy.deepcopy(radar.fields['differential_reflectivity'])
            zdr['data'] = zdr['data'] + 0.5
            radar.fields.update({'differential_reflectivity': zdr})

            ### Fixing calibration of Zh
            zh= copy.deepcopy(radar.fields['reflectivity'])
            zh['data'] = zh['data'] + 3
            radar.fields.update({'reflectivity': zh})

            # correcting for the antenna transition
            if radar.elevation['data'][0] > 90.0:
               radar.elevation['data'][0] = 0.0

            if radar.elevation['data'][1] > 90.0:
               radar.elevation['data'][1] = 0.0
        except:
            print("print could not read data")
            continue
        try: ## creating display object
            display = pyart.graph.RadarDisplay(radar)

            # Slant range
            #r = np.sqrt(np.power(display.x,2) + np.power(display.y,2))
            r = radar.gate_x['data']
            r_hyde = 64000
            margin = 5e3
            rmin = r_hyde - margin
            rmax = r_hyde + margin

            # Extracting radar variables
            ZH  = copy.deepcopy(display.fields['reflectivity'])
            ZH  = np.power(10.0,ZH['data']/10.0)
            ZDR = copy.deepcopy(display.fields['differential_reflectivity'])
            ZDR = np.power(10.0,ZDR['data']/10.0)
            KDP = copy.deepcopy(display.fields['specific_differential_phase'])
            KDP = KDP['data']
            RHO = copy.deepcopy(display.fields['cross_correlation_ratio'])
            RHO = RHO['data']
            DP  = copy.deepcopy(display.fields['differential_phase'])
            DP  = DP['data']

            ZH[r<=rmin] = np.NaN
            ZH[r>=rmax]  = np.NaN

            ZDR[r<=rmin] = np.NaN
            ZDR[r>=rmax]  = np.NaN

            KDP[r<=rmin] = np.NaN
            KDP[r>=rmax]  = np.NaN

            RHO[r<=rmin] = np.NaN
            RHO[r>=rmax]  = np.NaN

            diff_r = np.abs(r[0,:] - r_hyde)
            indx = diff_r.argmin()

            #hght = display.z[:,indx]
            hght = radar.gate_z['data'][:,indx]

            zh_vp[file_indx,:]  = np.interp(height, hght, 10*np.log10(np.nanmean(ZH,1)))
            zdr_vp[file_indx,:] = np.interp(height, hght, 10*np.log10(np.nanmean(ZDR,1)))
            kdp_vp[file_indx,:] = np.interp(height, hght, np.nanmean(KDP,1))
            rho_vp[file_indx,:] = np.interp(height, hght, np.nanmean(RHO,1))
            dp_vp[file_indx,:]  = np.interp(height, hght, np.nanmean(DP,1))

            #ObsTime.append(display.time_begin.isoformat())
            tstr = path.basename(filename)[0:12]
            ObsTime.append(datetime.strptime(tstr, '%Y%m%d%H%M').isoformat())
            file_indx = file_indx + 1
        except Exception as error:
            print(str(error))
            print("print could not plot data")
            raise error
            continue
    fname_supl = 'IKA_VP_from_RHI'
    time_filename = path.basename(filename)[0:8]
    fileOut = path.join(pathOut, time_filename + '_' + fname_supl + '.mat')
    print(fileOut)
    VP_RHI = {'ObsTime': ObsTime, 'ZH': zh_vp, 'ZDR': zdr_vp, 'KDP': kdp_vp, 'RHO': rho_vp, 'DP': dp_vp, 'height': height}
    sio.savemat(fileOut, {'VP_RHI':VP_RHI})


if __name__ == '__main__':
    #pathOut = '/Users/moiseev/Data/VP_RHI/'
    #pathIn = "/Volumes/uhradar/IKA_final/20140318/"
    home = path.expanduser('~')
    pathIn = path.join(home, 'DATA', 'IKA', '20180818RHI')
    pathOut = path.join(home, 'results', 'radcomp', 'vertical', 'vp_dmitri')
    main(pathIn, pathOut)

