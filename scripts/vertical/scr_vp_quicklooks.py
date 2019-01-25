# coding: utf-8
"""script for saving quicklooks of VP data"""

from os import path
from datetime import timedelta
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt

from radcomp import USER_DIR
from radcomp.vertical import case, multicase, RESULTS_DIR
from j24 import ensure_join


interactive = False


def plot_quicklooks(cases_iterator, save=True, saveid='everything',
                    params=None, savedir=None, **kws):
    """Plot and save quicklooks."""
    params = params or ['zh', 'zdr', 'kdp', 'RHO']
    if save:
        savedir = savedir or ensure_join(RESULTS_DIR, 'quicklooks', saveid)
    for caseid, c in cases_iterator:
        print(caseid)
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


def write_dates(datelistfile, casesname):
    """write start,end csv based on date list file"""
    dates = datetime_df(datelistfile)
    casesfile = path.join(USER_DIR, 'cases', casesname + '.csv')
    dates.to_csv(casesfile, index=None, date_format='%Y-%m-%d %H:%M')


def iterate_cases(casesname):
    """cases iterator from a case list"""
    cases = multicase.read_cases(casesname)
    return cases.case.iteritems()


def iterate_mat2case(datadir, fname_glob='*.mat'):
    """iterator over case objects from data files in a directory"""
    datafiles = glob(path.join(datadir, fname_glob))
    for datafile in datafiles:
        cid = path.basename(datafile)[:8]
        try:
            c = case.Case.from_mat(datafile)
            yield cid, c
        except ValueError as e:
            print(cid, e)


if __name__ == '__main__':
    plt.close('all')
    #params = ['ZH', 'ZDR', 'KDP', 'RHO']
    params = None
    #casesname = 'snow'
    #datelistfile = path.join(case.DATA_DIR, 'date.list')
    #write_dates(datelistfile, 'daily_quicklooks')
    #iterator = iterate_cases(casesname)
    datadir = path.expanduser('~/DATA/vprhi2')
    savedir = ensure_join(datadir, 'quicklooks', 'viridis')
    iterator = iterate_mat2case(datadir)#, fname_glob='201501*.mat')
    if interactive:
        plt.ion()
        save = False
    else:
        plt.ioff()
        save = True
    plot_quicklooks(iterator, save=save, params=params, cmap='viridis',
                    savedir=savedir)

