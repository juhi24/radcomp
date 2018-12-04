# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib as mpl
import matplotlib.pyplot as plt


class CasesPlotter:
    def __init__(self, cases):
        self.cases = cases
        self.fig = None
        self.axarr = None
        self.i_active = 0
        self.plot_kws = {}

    @property
    def n_cases(self):
        return self.cases.shape[0]

    def next_case(self):
        if self.i_active > self.n_cases-2:
            return self.i_active
        self.i_active += 1
        return self.i_active

    def prev_case(self):
        if self.i_active < 1:
            return 0
        self.i_active -= 1
        return self.i_active

    def active_case(self, i=None):
        i = i or self.i_active
        return self.cases.case.iloc[i]

    def plot(self, **kws):
        self.plot_kws = kws
        self.fig, self.axarr = self.active_case().plot(**kws)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

    def press(self, event):
        print(event.key)
        if event.key == 'right':
            self.next_case()
        elif event.key == 'left':
            self.prev_case()
        plt.close(self.fig)
        self.plot(**self.plot_kws)