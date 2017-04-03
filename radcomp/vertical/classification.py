# coding: utf-8
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def train(data, n_eigens, quiet=False, reduced=False, **kws):
    metadata = dict(fields=data.items.values, hmax=data.minor_axis.max())
    data_df = learn.pn2df(data)
    pca = pca_fit(data_df, n_components=n_eigens)
    if not quiet:
        learn.pca_stats(pca)
    if reduced:
        km = kmeans_pca_reduced(data_df, pca, **kws)
    else:
        km = kmeans_pca_init(data_df, pca, **kws)
    return pca, km, metadata


def pca_fit(data_df, whiten=True, **kws):
    pca = decomposition.PCA(whiten=whiten, **kws)
    pca.fit(data_df)
    return pca


def kmeans_pca_init(data_df, pca):
    km = KMeans(init=pca.components_, n_clusters=pca.n_components, n_init=1)
    km.fit(data_df)
    return km


def kmeans_pca_reduced(data_df, pca, n_clusters=20):
    km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    reduced = pca.transform(data_df)
    km.fit(reduced)
    return km


def classify(data_scaled, km):
    data_df = learn.pn2df(data_scaled)
    return pd.Series(data=km.predict(data_df), index=data_scaled.major_axis)


def load(name):
    with open(model_path(name), 'rb') as f:
        return pickle.load(f)


def plot_reduced(data, n_clusters):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    reduced_data = decomposition.PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(reduced_data) 
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Vega20,
               aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())


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

    def train(self, data, n_eigens=None, **kws):
        if n_eigens is None:
            n_eigens = self._n_eigens
        pca, km, metadata = train(data, n_eigens, **kws)
        self.pca = pca
        self.km = km
        self.params = metadata['fields']
        self.hmax = metadata['hmax']

