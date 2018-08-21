# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt
from scr_class_stats import init_rain, cl_frac_in_case, frac_in_case_hist


def proc_frac(cases, lcl, frac=True):
    """fraction or sum of process occurrences per case"""
    cl_sum = cases.case.apply(lambda x: 0)
    for cl in lcl:
        cl_sum += cases.case.apply(lambda x: cl_frac_in_case(x, cl, frac=False))
    if frac:
        sizes = cases.case.apply(lambda x: x.classes.size)
        return cl_sum/sizes
    return cl_sum





if __name__ == '__main__':
    plt.close('all')
    #cases_r, cc_r = init_rain()
    cl_hm = (10, 13)
    cl_dend = (11, 12, 14, 15)
    fracs = proc_frac(cases_r, cl_dend, frac=True)
    ax = frac_in_case_hist(cases_r, frac_per_case=fracs)


