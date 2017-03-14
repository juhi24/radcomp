# coding: utf-8
import locale
from os import path

locale.setlocale(locale.LC_ALL, 'C')
HOME = path.expanduser('~')
USER_DIR = path.join(HOME, '.radcomp')
RESULTS_DIR = path.join(HOME, 'results', 'radcomp')