# coding: utf-8
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gg = cc_r.data_above_ml['T'].T.groupby(cc_r.classes())
    tmed = gg.median().T
    tmed.plot()
    plt.grid('on')
    q1 = cc_r.data_above_ml['T'].T.quantile(q=0.25)
    q3 = cc_r.data_above_ml['T'].T.quantile(q=0.75)
    #  -3C:  400.. 700 m
    #  -8C: 1300..1600 m
    # -10C: 1600..1950 m
    # -20C: 3150..3600 m