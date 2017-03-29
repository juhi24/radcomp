# coding: utf-8
import numpy as np
#import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, RESULTS_DIR
from j24 import ensure_dir

plt.ioff()
plt.close('all')
np.random.seed(0)
n_comp = 20
scheme = '2014rhi_{n}comp'.format(n=n_comp)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'classes_summary', scheme))

cases = case.read_cases('training')
c = case.Case.by_combining(cases)
c.load_classification(scheme)
df = c.pcolor_classes()
for i, fig in df.fig.iteritems():
    fig.savefig(path.join(results_dir, 'class{:02d}.png'.format(i)))
