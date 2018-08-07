# coding: utf-8
"""script for saving quicklooks of VP data"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
import matplotlib.pyplot as plt
from os import path
from datetime import timedelta
from radcomp import USER_DIR
from radcomp.vertical import case, multicase
from j24 import ensure_join


interactive = True


def plot_quicklooks(cases, save=True, **kws):
    """Plot and save quicklooks."""
    params = ['ZH', 'zdr', 'kdp', 'RHO']
    if save:
        savedir = ensure_join(case.DATA_DIR, 'quicklooks_viridis')
    for _, c in cases.case.iteritems():
        fig, _ = c.plot(params=params, plot_fr=False, plot_t=True,
                        plot_azs=False, plot_snd=False, **kws)
        if save:
            filename = path.join(savedir, c.name()+'.png')
            fig.savefig(filename, bbox_inches='tight')
            plt.close(fig)


def datetime_df(datelistfile):
    """read dates from file and return in start,end DataFrame"""
    df = pd.read_csv(datelistfile, header=None, parse_dates=[0],
                     names=['start'])
    df['end'] = df.start + timedelta(days=1) - timedelta(minutes=1)
    return df


def write_dates(datelistfile):
    """write start,end csv based on date list file"""
    dates = datetime_df(datelistfile)
    dates.to_csv(casesfile, index=None, date_format='%Y-%m-%d %H:%M')


if __name__ == '__main__':
    plt.close('all')
    #casesname = 'daily_quicklooks'
    casesname = 'mlt_test'
    casesfile = path.join(USER_DIR, 'cases', casesname + '.csv')
    datelistfile = path.join(case.DATA_DIR, 'date.list')
    #write_dates(datelistfile)
    cases = multicase.read_cases(casesname)
    if interactive:
        plt.ion()
        save = False
    else:
        plt.ioff()
        save = True
    plot_quicklooks(cases, save=save, cmap='viridis')

