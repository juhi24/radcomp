# coding: utf-8
"""inverse transform experiments for selecting number of PCA components"""

import matplotlib.pyplot as plt

from radcomp.vertical import multicase, case, plotting

import conf


def load_case(rain=True):
    cases_name = 't_model' if rain else 'feb15'
    scheme_id = conf.SCHEME_ID_MELT if rain else conf.SCHEME_ID_SNOW
    cases = multicase.read_cases(cases_name)
    c = cases.case[0]
    c.load_classification(scheme_id)
    return c


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    c = load_case(rain=False)
    c.plot(cmap='viridis', plot_silh=False, above_ml_only=True)
    c.plot(cmap='viridis', plot_silh=False, inverse_transformed=True)
    # difference
    ref = c.cl_data.transpose(0,2,1)
    tr = c.inverse_transform()
    diff = (case.fillna(ref).subtract(tr))
    fig, axarr = plotting.plotpn(diff.abs())
    axarr[0].set_title('Absolute error of inverse transformed data')
    rmse = diff.pow(2).mean().mean().pow(0.5)
