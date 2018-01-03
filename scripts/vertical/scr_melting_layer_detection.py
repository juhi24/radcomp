# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from radcomp.vertical import case


def low_z(z):
    return


def low_rhohv(rhohv):
    return rhohv < 0.95


def melting_px_candidate(pn):
    return low_rhohv(pn['RHO'])


if __name__ == '__main__':
    plt.close('all')
    cases = case.read_cases('melting')
    c = cases.case.iloc[0]
    c_snow = cases.case.iloc[1] # no melting
    #c_snow.plot(params=['ZH', 'zdr', 'kdp', 'RHO'], cmap='viridis')
    rhohv = c.data['RHO']
    rhohv_snow = c_snow.data['RHO']
    rho_sample = rhohv.iloc[:,20]
    rho_snow_sample = rhohv_snow.iloc[:,20]
    c.data['MLT'] = melting_px_candidate(c.data)
    fig, axarr = c.plot(params=['ZH', 'zdr', 'kdp', 'RHO', 'MLT'], cmap='viridis')


