# coding: utf-8

import datetime

import pandas as pd


if __name__ == '__main__':
    diffs = cc.timestamps().diff()
    casegaps = diffs[diffs>cc.timedelta]
    gap = datetime.timedelta(hours=12)
    echo_gaps = diffs.copy()
    no_echo = cc.classes()==0
    i_no_echo = no_echo[no_echo].index
    echo_gaps.drop(i_no_echo, inplace=True)


