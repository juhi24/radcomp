# coding: utf-8
"""Just load the cases data and classification."""

from radcomp.vertical import multicase

import conf


if __name__ == '__main__':
    cases_id = 'rain'
    rain_season = cases_id in ('rain',)
    flag = 'ml_ok' if rain_season else None
    cc = multicase.MultiCase.from_caselist(cases_id, filter_flag=flag, has_ml=rain_season)
    name = conf.SCHEME_ID_RAIN if rain_season else conf.SCHEME_ID_SNOW
    cc.load_classification(name)