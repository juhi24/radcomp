# coding: utf-8
"""plot wrappers"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

SAVE_DEFAULT=True


def plot_cluster_centroids(c, **kws):
    c.plot_cluster_centroids()