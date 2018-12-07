# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import matplotlib.pyplot as plt

from radcomp.vertical import plotting


class CasesBase:
    """"""
    def __init__(self, cases):
        self.cases = cases


class CasesPlotter(CasesBase):
    """Interactively visualize a DataFrame of cases"""

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        self.fig = None
        self.axarr = None
        self.i_active = 0
        self.plot_kws = {}
        self.press = self.press_nav

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
        self.fig.canvas.mpl_connect('key_press_event', self.press)

    def press_nav(self, event):
        """key press logic"""
        print(event.key)
        if event.key == 'right':
            self.next_case()
        elif event.key == 'left':
            self.prev_case()
        plt.close(self.fig)
        self.plot(**self.plot_kws)


class ProfileMarker(CasesPlotter):
    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)

    def _select(self, xdata):
        try:
            dt = plotting.num2tstr(xdata)
        except AttributeError: # clicked outside axes
            return
        dt_nearest = self.active_case().nearest_datetime(dt)
        return dt_nearest.strftime(plotting.DATETIME_FMT_CSV)

    def click_select(self, event):
        tstr = self._select(event.xdata)
        print('{},'.format(tstr), end='')

    def release_select(self, event):
        tstr = self._select(event.xdata)
        print(tstr)

    def plot(self, **kws):
        super().plot(**kws)
        self.fig.canvas.mpl_connect('button_press_event', self.click_select)
        self.fig.canvas.mpl_connect('button_release_event', self.release_select)