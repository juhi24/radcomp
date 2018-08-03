# coding: utf-8
"""Test ML detection performance."""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pytest
from radcomp.vertical import multicase


def ml_top_ok(c, t_start, t_end, true_ml_top, max_error=300):
    """ML top test template."""
    top = c.ml_limits()[1]
    test_tops = top.loc[t_start:t_end]
    near_true_ml = abs(test_tops-true_ml_top)<max_error
    return near_true_ml.all()


@pytest.fixture
def testcases():
    """Prepare test cases."""
    cases = multicase.read_cases('mlt_test')
    for _, c in cases.case.iteritems():
        c.load_classification('mlt_18eig17clus_pca')
    return cases


## TESTS

def test_140612(testcases):
    """Test if detected ML top height is appropriate."""
    c = testcases.case['140612T03']
    assert ml_top_ok(c, '2014-06-12 03:13:00', '2014-06-12 08:59:00', 3000,
                     max_error=350)


def test_140729(testcases):
    """Test if detected ML top height is appropriate."""
    c = testcases.case['140729T09']
    assert ml_top_ok(c, '2014-07-29 10:13:00', '2014-07-29 10:29:00', 2900)
