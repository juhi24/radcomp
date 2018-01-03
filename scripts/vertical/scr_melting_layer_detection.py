# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import case


def dbz_is_low(dbz):
    return dbz < 1


def rhohv_is_low(rhohv):
    return rhohv < 0.97


def melting_px_candidate(pn):
    low_rhohv = rhohv_is_low(pn['RHO'])
    low_dbz = dbz_is_low(pn['ZH'])
    return low_rhohv & -low_dbz


if __name__ == '__main__':
    plt.close('all')
    cases = case.read_cases('melting')
    c = cases.case.iloc[0]
    c_snow = cases.case.iloc[1] # no melting
    rhohv = c.data['RHO']
    rhohv_snow = c_snow.data['RHO']
    rho_sample = rhohv.iloc[:,20]
    rho_snow_sample = rhohv_snow.iloc[:,20]
    c.data['MLT'] = melting_px_candidate(c.data)
    c_snow.data['MLT'] = melting_px_candidate(c_snow.data)
    fig, axarr = c.plot(params=['ZH', 'zdr', 'kdp', 'RHO', 'MLT'], cmap='viridis')
    c_snow.plot(params=['ZH', 'zdr', 'kdp', 'RHO', 'MLT'], cmap='viridis')


