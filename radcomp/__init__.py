# coding: utf-8
"""radcomp"""

import locale
from os import path
from j24 import home, ensure_join
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

locale.setlocale(locale.LC_ALL, 'C')
HOME = home()
USER_DIR = path.join(HOME, '.radcomp')
RESULTS_DIR = ensure_join(HOME, 'results', 'radcomp')
CACHE_DIR = ensure_join(HOME, '.cache', 'radcomp')
CACHE_TMP_DIR = ensure_join(CACHE_DIR, 'tmp')
