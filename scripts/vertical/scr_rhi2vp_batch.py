# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import pandas as pd
from os import path

if __name__ == '__main__':
    listfile = path.expanduser('~/ika_rhi.list')
    files = pd.read_csv(listfile, header=['filepath'])
    files['basename'] = files.filepath.apply(path.basename)
    files['dirname'] = files.filepath.apply(path.dirname)