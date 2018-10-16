# coding: utf-8

from os import path

from radcomp.vertical import case

datadir = path.expanduser('~/DATA/vprhi2')
filu = path.join(datadir, '20150331_IKA_vprhi.mat')
params = ['ZH', 'ZDR', 'KDP', 'RHO']
c = case.Case.from_mat(filu)
c.plot(params=params, plot_fr=False, plot_t=True, plot_azs=False,
       plot_snd=False)