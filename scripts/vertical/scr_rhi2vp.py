# coding: utf-8
"""Extract vertical profiles over Hyytiälä from IKA RHI.

Authors: Dmitri Moisseev and Jussi Tiira
"""

from os import path
from radcomp.tools.rhi import rhi2vp


if __name__ == '__main__':
    #pathOut = '/Users/moiseev/Data/VP_RHI/'
    #pathIn = "/Volumes/uhradar/IKA_final/20140318/"
    home = path.expanduser('~')
    pathIn = path.join(home, 'DATA', 'IKA', '20180818RHI')
    pathOut = path.join(home, 'results', 'radcomp', 'vertical', 'vp_dmitri')
    rhi2vp(pathIn, pathOut)

