# coding: utf-8
"""Dumb VPR model development"""

import matplotlib.pyplot as plt


def vpr_median(cc_r, km_above_ml=1100):
    """vpr diffs based on median ze above ml"""
    z = cc_r.data.zh.iloc[0, :]
    zt = cc_r.cl_data.zh.loc[:, km_above_ml]
    cl = cc_r.classes()
    mz = z.groupby(cl).median()
    mzt = zt.groupby(cl).median()
    return mz-mzt



if __name__ == '__main__':
    plt.ion()
    vpr = vpr_median(cc_r)
