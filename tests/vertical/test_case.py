# coding: utf-8

import pytest
import matplotlib.pyplot as plt
import pandas as pd
from radcomp.vertical import multicase, classification


def class_scheme_name():
    """generate test class scheme name"""
    return classification.scheme_name(
            basename='14-16',
            n_eigens=19,
            n_clusters=19,
            reduced=True,
            use_temperature=True,
            t_weight_factor=0.8,
            radar_weight_factors=dict(zdr=0.5))


def cases_from_cases_set():
    """setup cases based on a cases list"""
    return multicase.read_cases('debug')


def case_post(c):
    """finalize case setup"""
    scheme_name = class_scheme_name()
    c.load_classification(scheme_name)
    c.load_pluvio()


@pytest.fixture
def case_full_setup():
    """case setup with classification"""
    c = cases_from_cases_set().case.iloc[0]
    case_post(c)
    return c


## TESTS

def test_mean_delta(case_full_setup):
    """mean delta should be 15 minutes by default"""
    assert case_full_setup.mean_delta() == pd.Timedelta(minutes=15)


##

if __name__ == '__main__':
    scheme_name = class_scheme_name()
    plt.ion()
    plt.close('all')
    c = case_full_setup()
    #fig, axarr = c.plot(cmap='viridis')
    #fig, axarr = c.plot(plot_fr=False, plot_t=False)

