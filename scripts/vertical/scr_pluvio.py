# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

#import pandas as pd
import baecc.instruments.pluvio as pl
from os import path
from j24 import home

filename = path.join(home(), 'DATA', 'Pluvio400', 'pluvio400_01_2017020202.txt')
pluv = pl.Pluvio(filenames=[filename])
