# coding: utf-8
import numpy as np
import pandas as pd
from os import path
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from radcomp import learn, RESULTS_DIR


META_SUFFIX = '_metadata'

def model_path(name):
    return path.join(RESULTS_DIR, 'models', name + '.pkl')

def save_model(model, name):
    savepath = model_path(name)
    joblib.dump(model, savepath)
    return savepath

def save_data(data, name):
    joblib.dump(data, model_path(name + '_data'))
    hmax = np.ceil(data.minor_axis.max())
    fields = list(data.items)
    metadata = dict(hmax=hmax, fields=fields)
    joblib.dump(metadata, model_path(name + META_SUFFIX))

def load_model(name):
    loadpath = model_path(name)
    model = joblib.load(loadpath)
    return model

def save_classification(pca, kmeans, data, name):
    save_model(pca, name + '_pca')
    save_model(kmeans, name + '_kmeans')
    save_data(data, name)

def load_classification(name):
    '''return pca, km, metadata'''
    pca = load_model(name + '_pca')
    km = load_model(name + '_kmeans')
    metadata = joblib.load(model_path(name + META_SUFFIX))
    return pca, km, metadata

def train(data_scaled, n_eigens, quiet=False, **kws):
    data_df = learn.pn2df(data_scaled)
    pca = pca_fit(data_df, n_components=n_eigens)
    if not quiet:
        learn.pca_stats(pca)
    km = kmeans(data_df, pca)
    return pca, km

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

class VPC:
    """vertical profile classification scheme"""
    
    def __init__(self, pca=None, km=None, hmax=None, fields=None):
        self.pca = pca
        self.km = km # k means
        self.hmax = hmax
        self.fields = fields
        self.kdpmax = None

    @classmethod
    def from_pkl(cls, name):
        pca, km, metadata = load_classification(name)
        return cls(pca=pca, km=km, hmax=metadata['hmax'], fields=metadata['fields'])