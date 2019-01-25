# coding: utf-8

import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import case, classification, RESULTS_DIR
from j24 import ensure_dir
from warnings import warn

plt.ioff()
plt.close('all')

case_set = '14-16by_hand'
n_eigens = 25
n_clusters = 20
reduced = True
save = True

cases = case.read_cases(case_set)
name = classification.scheme_name(basename='14-16_t', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'z-lwe_comparison2', name, case_set))

def plot_ze_lwe_comparison(c):
    z=c.cl_data.ZH.iloc[:,1]
    i=c.lwe(offset_half_delta=False)
    iz=10*i**1.2
    ioz = 10*c.pluvio.intensity()**1.2
    fig = plt.figure()
    z.plot(drawstyle='steps')
    iz.plot(drawstyle='steps')
    #ioz.plot(drawstyle='steps')
    return fig

for i, c in cases.case.iteritems():
    print(i)
    try:
        c.load_classification(name)
        c.load_pluvio()
    except ValueError as e:
        warn(str(e))
        continue
    fig = plot_ze_lwe_comparison(c)
    fig.savefig(path.join(results_dir, c.name()+'.png'))
    plt.close(fig)

