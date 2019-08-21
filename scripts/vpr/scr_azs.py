# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

from radcomp import azs


if __name__ == '__main__':
    plt.ion()
    plt.close('all')
    data = azs.load_series()
    types = {'class': int, 'azs': float}
    azs_cl = pd.concat([cc_s.classes(), data], axis=1).dropna().astype(types)
    azs_cl.boxplot(column='class')
    g = azs_cl.groupby('class')


