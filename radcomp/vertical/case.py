# coding: utf-8
import numpy as np
import pandas as pd
import scipy.io
from os import path
from functools import partial
from radcomp.vertical import filtering, classification, plotting, NAN_REPLACEMENT
from radcomp import HOME

DATA_DIR = path.join(HOME, 'DATA', 'ToJussi')


def dt2path(dt, datadir):
    return path.join(datadir, dt.strftime('%Y%m%d_IKA_VP_from_RHI.mat'))


def vprhimat2pn(datapath):
    data = scipy.io.loadmat(datapath)['VP_RHI']
    fields = list(data.dtype.fields)
    fields.remove('ObsTime')
    fields.remove('height')
    str2dt = lambda tstr: pd.datetime.strptime(tstr,'%Y-%m-%dT%H:%M:%S')
    t = list(map(str2dt, data['ObsTime'][0][0]))
    h = data['height'][0][0][0]
    data_dict = {}
    for field in fields:
        data_dict[field] = data[field][0][0].T
    return pd.Panel(data_dict, major_axis=h, minor_axis=t)

def fname_range(dt_start, dt_end):
    dt_range = pd.date_range(dt_start.date(), dt_end.date())
    dt2path_map = partial(dt2path, datadir=DATA_DIR)
    return map(dt2path_map, dt_range)


def kdp2phidp(kdp, dr_km):
    kdp_filled = kdp.fillna(0)
    return 2*kdp_filled.cumsum().multiply(dr_km, axis=0)


def data_range(dt_start, dt_end):
    fnames = fname_range(dt_start, dt_end)
    pns = map(vprhimat2pn, fnames)
    return pd.concat(pns, axis=2).loc[:, :, dt_start:dt_end]


def prepare_pn(pn, kdpmax=0.5):
    dr = pd.Series(pn.major_axis.values, index=pn.major_axis).diff().bfill()
    dr_km = dr/1000
    pn_new = pn.copy()
    pn_new['KDP_orig'] = pn_new['KDP'].copy()
    pn_new['KDP'][pn_new['KDP']<0] = np.nan
    pn_new['phidp'] = kdp2phidp(pn_new['KDP'], dr_km)
    kdp = pn_new['KDP'] # a view
    kdp[kdp>kdpmax] = 0
    kdp[kdp<0] = 0
    pn_new = filtering.fltr_median(pn_new)
    pn_new = filtering.fltr_ground_clutter_median(pn_new)
    return pn_new


def dt2pn(dt0, dt1):
    pn_raw = data_range(dt0, dt1)
    return prepare_pn(pn_raw)


def fillna(dat, field=''):
    data = dat.copy()
    if isinstance(data, pd.Panel):
        for field in list(data.items):
            data[field].fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    elif isinstance(data, pd.DataFrame):
        data.fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    return data


def prepare_data(pn, fields=['ZH', 'ZDR', 'kdp'], hmax=10e3, kdpmax=None):
    data = pn[fields, 0:hmax, :].transpose(0,2,1)
    if kdpmax is not None:
        data['KDP'][data['KDP']>kdpmax] = np.nan
    return fillna(data)


def prep_data(pn, vpc):
    return prepare_data(pn, fields=vpc.fields, hmax=vpc.hmax, kdpmax=vpc.kdpmax)


def scale_data(pn):
    scaling_limits = {'ZH': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3), 'KDP': (0, 0.5), 
                      'kdp': (0, 0.15)}
    scaled = pn.copy()
    for field, data in scaled.iteritems():
        data -= scaling_limits[field][0]
        data *= 1.0/scaling_limits[field][1]
        scaled[field] = data
    return scaled


class Case:
    def __init__(self, data=None, cl_data=None, cl_data_scaled=None, classes=None):
        self.data = data
        self.cl_data = cl_data # classification data
        self.cl_data_scaled = cl_data_scaled
        self.classes = classes
        self.class_scheme = None

    @classmethod
    def from_dtrange(cls, t0, t1):
        pn = dt2pn(t0, t1)
        return cls(data=pn)

    def load_classification(self, name):
        self.class_scheme = classification.VPC.from_pkl(name)
        self.classify()

    def prepare_cl_data(self, save=True):
        if self.data is not None:
            cl_data = prep_data(self.data, self.class_scheme)
            if save:
                self.cl_data = cl_data
            return cl_data
        return None
        
    def scale_cl_data(self, save=True):
        """scaled version of classification data"""
        if self.cl_data is None:
            self.prepare_cl_data()
        if self.cl_data is not None:
            scaled = scale_data(self.cl_data)
            if save:
                self.cl_data_scaled = scaled
            return scaled
        return None

    def classify(self, save=True):
        """classify based on class_scheme"""
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        if self.cl_data_scaled is not None:
            classes = classification.classify(self.cl_data_scaled, self.class_scheme.km)
            if save:
                self.classes = classes
            return classes
        return None

    def plot_classes(self):
        return plotting.plot_classes(self.cl_data_scaled, self.classes)

    def plot(self, fields=None, **kws):
        if fields is None:
            fields = self.class_scheme.fields
        fig, axarr = plotting.plotpn(self.data, fields=fields, **kws)
        if self.classes is not None:
            for iax in range(len(axarr)-1):
                plotting.class_colors(self.classes, ax=axarr[iax])
        