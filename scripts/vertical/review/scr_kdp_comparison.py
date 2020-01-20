# coding: utf-8

from os import path

from radcomp.tools import rhi
from radcomp.vertical import case


ika_path = '/media/jussitii/04fafa8f-c3ca-48ee-ae7f-046cf576b1ee/IKA_final'


def genmat():
    dir_in = path.join(ika_path, 'tmp')
    out = path.join(ika_path, 'out')
    return rhi.mat_workflow(dir_in, out, kdp_debug=True)


if __name__ == '__main__':
    matname = '20140221_IKA_vprhi.mat'
    case.Case.from_mat(path.join(ika_path, 'out_v', matname))

