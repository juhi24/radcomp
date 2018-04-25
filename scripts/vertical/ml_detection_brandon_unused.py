# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type


def consecutive_grouper(s):
    """Consecutive values to have same integer -> 111223333455"""
    return (s != s.shift()).cumsum()


def _check_above_ml(ml, rho, grouper, iml, rholim):
    """check that value above expanded ml makes sense"""
    try:
        rho_above_ml = rho[grouper==iml+1].iloc[0]
    except IndexError:
        return ml == np.inf # all False
    if (rho_above_ml < rholim) or np.isnan(rho_above_ml):
        return ml == np.inf # all False
    return ml


def _expand_ml_series(ml, rho, rholim=RHO_MAX):
    """expand detected ml using rhohv"""
    grouper = consecutive_grouper(rho<rholim)
    selected = grouper[ml]
    if selected.empty:
        return ml
    iml = selected.iloc[0]
    ml_new = grouper==iml
    ml_new = _check_above_ml(ml_new, rho, grouper, iml, rholim)
    return ml_new


def expand_ml(ml, rho):
    """expand ML"""
    for t, mlc in ml.iteritems():
        ml[t] = _expand_ml_series(mlc, rho[t])
    return ml


def first_consecutive(s):
    """only first consecutive group of trues is kept"""
    grouper = consecutive_grouper(s)
    g = s.groupby(grouper)
    true_groups = g.mean()[g.mean()]
    if true_groups.empty:
        return grouper == -1
    return grouper == [true_groups.index[0]]


def detect_ml(mli, rho, mli_thres=MLI_THRESHOLD, rholim=RHO_MAX):
    """Detect ML using melting layer indicator."""
    ml = mli > mli_thres
    ml[rho>0.975] = False
    ml = ml.apply(first_consecutive)
    ml = expand_ml(ml, rho)
    return ml


def ml_top(ml, maxh=H_MAX, no_ml_val=np.nan):
    """extract ml top height from detected ml"""
    top = ml[::-1].idxmax()
    if no_ml_val is not None:
        top[top>maxh] = no_ml_val
    return top
