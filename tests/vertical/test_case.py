# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pytest
import matplotlib.pyplot as plt
import pandas as pd
from radcomp.vertical import case, classification


@pytest.fixture
def class_scheme_name():
    return classification.scheme_name(
            basename='14-16',
            n_eigens=19,
            n_clusters=19,
            reduced=True,
            use_temperature=True,
            t_weight_factor=0.8,
            radar_weight_factors=dict(zdr=0.5))


@pytest.fixture
def cases_from_cases_set():
    case_set = '14-16by_hand'
    return case.read_cases(case_set)


@pytest.fixture
def case_full_setup():
    scheme_name = class_scheme_name()
    c = cases_from_cases_set().case.loc['140221-22']
    c.load_classification(scheme_name)
    c.load_pluvio()
    return c


def test_mean_delta(case_full_setup):
    assert case_full_setup.mean_delta() == pd.Timedelta(minutes=15)


if __name__ == '__main__':
    scheme_name = class_scheme_name()
    plt.ion()
    plt.close('all')
    c = case_full_setup()
    #fig, axarr = c.plot(cmap='viridis')
    #fig, axarr = c.plot(plot_fr=False, plot_t=False)

