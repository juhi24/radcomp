# coding: utf-8
"""script for debugging ML"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import multicase

from conf import SCHEME_ID_MELT


name = 'mlt2_3eig10clus_pca'
#name = SCHEME_ID_MELT


if __name__ == '__main__':
    plt.close('all')
    cases = multicase.read_cases('ml_debug')
    c = cases.case[0]
    c.load_classification(name)
    c.plot(params=['ZH', 'zdr', 'RHO', 'MLI'], cmap='viridis')
