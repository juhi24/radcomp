# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot
from radcomp.vertical import case


def dbz_is_low(dbz):
    return dbz < 1


def rhohv_is_low(rhohv):
    return rhohv < 0.97


def drop_short_bool_sequence(df_bool, limit=5):
    """Drop sequences of true that are shorter than limit."""
    return df_bool.rolling(limit, center=True).sum() > (limit-0.5)


def melting_px_candidate(pn):
    low_rhohv = rhohv_is_low(pn['RHO'])
    low_dbz = dbz_is_low(pn['ZH'])
    melting = low_rhohv & -low_dbz
    melting = drop_short_bool_sequence(melting)
    return melting


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


