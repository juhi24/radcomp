# coding: utf-8
import matplotlib.pyplot as plt
from os import path
from radcomp.vertical import (case, casedf, insitu, classification, plotting,
                              RESULTS_DIR)
from j24 import ensure_dir

plt.ion()
#plt.close('all')

n_eigens = 19
n_clusters = 19
reduced = True
use_temperature = True
t_weight_factor = 0.8
radar_weight_factors = dict(zdr=0.5)

name = classification.scheme_name(basename='14-16', n_eigens=n_eigens,
                                  n_clusters=n_clusters, reduced=reduced,
                                  use_temperature=use_temperature,
                                  t_weight_factor=t_weight_factor,
                                  radar_weight_factors=radar_weight_factors)
results_dir = ensure_dir(path.join(RESULTS_DIR, 'class_scatter', name))
table = dict(density=insitu.TABLE_FILTERED_PKL, intensity=insitu.TABLE_PKL)
cases = case.read_cases('14-16by_hand')

#g = pd.read_pickle(table['intensity'])
classes_all = casedf.classes_comb(cases, name)
rate_all = casedf.lwe_comb(cases)
#lwp_all = casedf.lwp_comb(cases)
t_all = casedf.t_comb(cases)
fr_all = casedf.fr_comb(cases)
mapping = cases.case.iloc[0].class_color_mapping()
default_color = (0, 0, 0)
fig_rate = plotting.hists_by_class(rate_all, classes_all, mapping=mapping,
                                   default=default_color)
fig_fr = plotting.hists_by_class(fr_all, classes_all, mapping=mapping,
                                 default=default_color)
#fig_lwp = plotting.hists_by_class(lwp_all, classes_all)
fig_t = plotting.hists_by_class(t_all, classes_all, mapping=mapping,
                                default=default_color)
fig_rate.savefig(path.join(results_dir, rate_all.name + '.png'))
fig_fr.savefig(path.join(results_dir, fr_all.name + '.png'))
#fig_lwp.savefig(path.join(results_dir, lwp_all.name + '.png'))
fig_t.savefig(path.join(results_dir, t_all.name + '.png'))

