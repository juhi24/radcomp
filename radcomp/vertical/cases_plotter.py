# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib as mpl
import matplotlib.pyplot as plt


class CasesPlotter:
    """Interactively visualize a DataFrame of cases"""

    def __init__(self, cases):
        self.cases = cases
        self.fig = None
        self.axarr = None
        self.i_active = 0
        self.plot_kws = {}

    @property
    def n_cases(self):
        """number of cases"""
        return self.cases.shape[0]

    def next_case(self):
        """Set next case as active."""
        if self.i_active > self.n_cases-2:
            return self.i_active
        self.i_active += 1
        return self.i_active

    def prev_case(self):
        """Set previous case as active."""
        if self.i_active < 1:
            return 0
        self.i_active -= 1
        return self.i_active

    def active_case(self, i=None):
        """Get and optionally set active case."""
        if i is None:
            i = self.i_active
        else:
            self.i_active = i
        return self.cases.case.iloc[i]

    def plot(self, **kws):
        """Plot current active case."""
        self.plot_kws = kws
        self.fig, self.axarr = self.active_case().plot(**kws)
        self.fig.canvas.mpl_connect('key_press_event', self._press)

    def _press(self, event):
        """key press logic"""
        print(event.key)
        if event.key == 'right':
            self.next_case()
        elif event.key == 'left':
            self.prev_case()
        plt.close(self.fig)
        self.plot(**self.plot_kws)