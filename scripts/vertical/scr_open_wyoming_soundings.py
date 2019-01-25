# coding: utf-8
"""open soundings from wyoming website"""

import webbrowser
import time
import pandas as pd
from radcomp import sounding
from radcomp.vertical import multicase
from conf import SCHEME_ID_MELT


case_set = 'melting'
name = SCHEME_ID_MELT


if __name__ == '__main__':
    search_class = 11
    mc = multicase.MultiCase.from_caselist(case_set, filter_flag='ml_ok')
    mc.load_classification(name)
    matchi = mc.classes[mc.classes==search_class].index
    matcha = pd.Series(data=matchi, index=matchi)
    match = matcha.apply(sounding.round_hours, hres=12).drop_duplicates()
    for i, t in match.iteritems():
        url = sounding.sounding_url(t, dtype='text')
        webbrowser.open(url)
        time.sleep(1)


