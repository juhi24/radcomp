# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

#import pandas as pd
import baecc.instruments.pluvio as pl
from os import path
from j24 import home

filename2 = path.join(home(), 'DATA', 'Pluvio400', 'pluvio400_01_2016061807.txt')
filename1 = path.join(home(), 'DATA', 'Pluvio400', 'pluvio400_01_2016061806.txt')
p4 = pl.Pluvio(filenames=[filename1, filename2])

filename2 = path.join(home(), 'DATA', 'Pluvio200', 'pluvio200_02_2016061807.txt')
filename1 = path.join(home(), 'DATA', 'Pluvio200', 'pluvio200_02_2016061806.txt')
p2 = pl.Pluvio(filenames=[filename1, filename2])

p2.acc().plot(drawstyle='steps')
p4.acc().plot(drawstyle='steps')