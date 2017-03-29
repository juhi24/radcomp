# coding: utf-8
import numpy as np
#import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, RESULTS_DIR
from j24 import ensure_dir

plt.ion()
plt.close('all')
np.random.seed(0)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary'))
n_comp = 20
scheme = '2014rhi_{n}comp'.format(n=n_comp)

cases = case.read_cases('training')
c = case.Case.by_combining(cases)
c.load_classification(scheme)
c.pcolor_classes()
