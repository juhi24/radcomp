# coding: utf-8
import pickle
#import numpy as np
import pandas as pd
from os import path
from sklearn import decomposition
from sklearn.cluster import KMeans
from radcomp import learn, USER_DIR
from j24 import ensure_dir


META_SUFFIX = '_metadata'
MODEL_DIR = ensure_dir(path.join(USER_DIR, 'class_schemes'))

def model_path(name):
    """/path/to/classification_scheme_name.pkl"""
    return path.join(MODEL_DIR, name + '.pkl')


def train(data, n_eigens, quiet=False):
    metadata = dict(fields=data.items.values, hmax=data.minor_axis.max())
    data_df = learn.pn2df(data)
    pca = pca_fit(data_df, n_components=n_eigens)
    if not quiet:
        learn.pca_stats(pca)
    km = kmeans(data_df, pca)
    return pca, km, metadata


def pca_fit(data_df, whiten=True, **kws):
    pca = decomposition.PCA(whiten=whiten, **kws)
    pca.fit(data_df)
    return pca


def kmeans(data_df, pca):
    km = KMeans(init=pca.components_, n_clusters=pca.n_components, n_init=1)
    km.fit(data_df)
    return km


def classify(data_scaled, km):
    data_df = learn.pn2df(data_scaled)
    return pd.Series(data=km.predict(data_df), index=data_scaled.major_axis)


def load(name):
    with open(model_path(name), 'rb') as f:
        return pickle.load(f)


class VPC:
    """vertical profile classification scheme"""
    
    def __init__(self, pca=None, km=None, hmax=None, params=None, n_eigens=None):
        self.pca = pca
        self.km = km # k means
        self.hmax = hmax
        self.params = params
        self.kdpmax = None
        self.data = None # training data
        self._n_eigens = n_eigens

    @classmethod
    def by_training(cls, data, n_eigens):
        pca, km, metadata = train(data, n_eigens)
        return cls.using_metadict(metadata, pca=pca, km=km)

    @classmethod
    def using_metadict(cls, metadata, **kws):
        return cls(hmax=metadata['hmax'], params=metadata['fields'], **kws)

    @classmethod
    def load(cls, name):
        obj = load(name)
        if isinstance(obj, cls):
            return obj
        raise Exception('Not a {} object.'.format(cls))

    def save(self, name):
        with open(model_path(name), 'wb') as f:
            pickle.dump(self, f)

    def train(self, data, n_eigens=None):
        if n_eigens is None:
            n_eigens = self._n_eigens
        pca, km, metadata = train(data, n_eigens)
        self.pca = pca
        self.km = km
        self.params = metadata['fields']
        self.hmax = metadata['hmax']
