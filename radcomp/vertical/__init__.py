# coding: utf-8
from os import path

home = path.expanduser('~')
NAN_REPLACEMENT = {'ZH': -10, 'ZDR': 0, 'KDP': 0}
RESULTS_DIR = path.join(home, 'results', 'radcomp', 'vertical')

from radcomp.vertical.tools import m2km
