# coding: utf-8
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from collections import OrderedDict
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from radcomp import learn, USER_DIR
from j24 import ensure_dir, limitslist
from j24.learn import pca_stats


META_SUFFIX = '_metadata'
MODEL_DIR = ensure_dir(path.join(USER_DIR, 'class_schemes'))

def weight_factor_str(param, value):
    out = '_' + param
    if value != 1:
        out += str(value).replace('.', '')
    return out

def scheme_name(basename='baecc+1415', n_eigens=30, n_clusters=20,
                reduced=True, use_temperature=False, t_weight_factor=1,
                radar_weight_factors=None):
    if reduced:
        qualifier = '_pca'
    else:
        qualifier = ''
    if use_temperature:
        basename += weight_factor_str('t', t_weight_factor)
    if radar_weight_factors:
        for field, factor in radar_weight_factors.items():
            if factor==1:
                continue
            basename += weight_factor_str(field, factor)
    schemefmt = '{base}_{neig}eig{nclus}clus{qualifier}'
    return schemefmt.format(base=basename, neig=n_eigens, nclus=n_clusters,
                            qualifier=qualifier)

def model_path(name):
    """/path/to/classification_scheme_name.pkl"""
    return path.join(MODEL_DIR, name + '.pkl')


def train(data_df, pca, quiet=False, reduced=False, n_clusters=20):
    if not quiet:
        pca_stats(pca)
    if reduced:
        km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=40, n_jobs=-1)
    else:
        km = KMeans(init=pca.components_, n_clusters=pca.n_components, n_init=1)
    classes_arr = km.fit_predict(data_df)
    return km, classes_arr


def pca_fit(data_df, whiten=False, **kws):
    pca = decomposition.PCA(whiten=whiten, **kws)
    pca.fit(data_df)
    return pca


def load(name):
    with open(model_path(name), 'rb') as f:
        return pickle.load(f)


def sort_by_column(arr, by=0):
    """sort array by column"""
    df = pd.DataFrame(arr)
    df_sorted = df.sort_values(by=by)
    mapping = pd.Series(index=df_sorted.index, data=range(df_sorted.shape[0]))
    return df_sorted.values, mapping.sort_index()


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
               cmap=plt.cm.tab20,
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
    """
    vertical profile classification scheme

    Attributes:
        pca (sklearn.decomposition.PCA): for VP dimension reduction
        km (sklearn.cluster.KMeans): for clustering
        hlimits (float, float): (hmin, hmax)
        params (array_like): identifier names of polarimetric radar variables
        params_extra (array_like): identifier names of additional clustering variables
        reduced (bool): if True, dimension reduction is used
        kdpmax
        data
        extra_weight_factor (float, optional): temperature weight factor
            in clustering, 1 if unspecified
        radar_weight_factors (dict of {str: float} pairs, optional):
            radar variable relative weight factors in clustering, 1 if unspecified
        basename (str): base name for the scheme id
    """

    def __init__(self, pca=None, km=None, hlimits=None, params=None,
                 reduced=False, n_eigens=None, n_clusters=None,
                 t_weight_factor=1, radar_weight_factors=None, basename=None,
                 use_temperature=False):
        self.pca = pca
        self.km = km  # k means
        self.hlimits = hlimits
        self.params = params
        self.params_extra = []
        self.reduced = reduced
        self.kdpmax = None
        self.data = None  # training or classification data
        self.extra_weight_factor = t_weight_factor
        self.radar_weight_factors = radar_weight_factors
        self.basename = basename
        self.use_temperature = use_temperature
        self.mapping = None
        self.training_data = None # archived training data
        self.training_result = None # archived training data classes
        self._n_eigens = n_eigens
        self._n_clusters = n_clusters

    def __repr__(self):
        return '<VPC {}>'.format(self.name())

    def __str__(self):
        return self.name()

    @property
    def n_clusters(self):
        return self._n_clusters or self.km.n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        self._n_clusters = n_clusters
        try:
            self.km.n_clusters = n_clusters
        except AttributeError:
            pass

    @classmethod
    def using_metadict(cls, metadata, **kws):
        return cls(hlimits=metadata['hlimits'], params=metadata['fields'], **kws)

    @classmethod
    def load(cls, name):
        """VPC object by loading a pickle"""
        obj = load(name)
        if isinstance(obj, cls):
            return obj
        raise Exception('Not a {} object.'.format(cls))

    def name(self):
        """scheme name string"""
        if self.basename is None:
            raise TypeError('basename is not set')
        return scheme_name(basename=self.basename, n_eigens=self._n_eigens,
                           n_clusters=self.n_clusters,
                           reduced=self.reduced,
                           use_temperature=self.use_temperature,
                           t_weight_factor=self.extra_weight_factor,
                           radar_weight_factors=self.radar_weight_factors)

    def get_class_list(self):
        """class number range"""
        return range(self.n_clusters)

    def save(self, **kws):
        """Save scheme to default location in pickle format."""
        with open(model_path(self.name(**kws)), 'wb') as f:
            pickle.dump(self, f)

    def train(self, data=None, n_eigens=None, extra_df=None, **kws):
        """Perform clustering of the training data to initialize classes."""
        if n_eigens is None:
            n_eigens = self._n_eigens
        if data is None:
            training_data = self.data
        else:
            training_data = self.prepare_data(data, n_components=n_eigens,
                                              extra_df=extra_df)
        self.training_data = training_data
        self.km, cl_arr = train(training_data, self.pca, reduced=self.reduced,
                                n_clusters=self.n_clusters, **kws)
        self.training_result = self.prep_classes(cl_arr, training_data.index)

    def prep_classes(self, cl_arr, index):
        """Map classes as Series"""
        self.map_components()
        classes = self.mapping[cl_arr].values
        return pd.Series(data=classes, index=index)

    def classify(self, data_scaled, **kws):
        """classify scaled observations"""
        data = self.prepare_data(data_scaled, **kws)
        cl_arr = self.km.predict(data)
        classes = self.prep_classes(cl_arr, data.index)
        tr = self.training_result
        td = self.training_data
        training_classes = tr.loc[tr.index.difference(classes.index)].copy()
        training_data = td.loc[td.index.difference(data.index)].copy()
        cl_silh = pd.concat((training_classes, classes))
        data_silh = pd.concat((training_data, data))
        silh = silhouette_samples(data_silh, cl_silh)
        silh = pd.Series(data=silh, index=cl_silh.index).loc[classes.index]
        return classes, silh

    def map_components(self):
        """Set class mapping, return sorted components and extra parameters"""
        centroids = self.km.cluster_centers_
        n_extra = len(self.params_extra)
        if n_extra < 1:
            components = centroids
            extra = []
        else:
            components = centroids[:, :-n_extra]
            extra = centroids[:, -n_extra:]/self.extra_weight_factor
        components, self.mapping = sort_by_column(components, by=0)
        extra_df = pd.DataFrame(extra, columns=self.params_extra)
        if not extra_df.empty:
            extra_df = extra_df.loc[self.mapping.sort_values().index]
            extra_df.reset_index(drop=True, inplace=True)
        return components, extra_df

    def clus_centroids_df(self):
        """
        cluster centroids DataFrame, extra parameters in separate DataFrame
        """
        components, extra_df = self.map_components()
        if self.reduced:
            centroids = self.pca.inverse_transform(components)
        return pd.DataFrame(centroids.T), extra_df

    def clus_centroids_pn(self):
        """
        cluster centroids Panel, extra parameters in separate DataFrame
        """
        clus_centroids, extra = self.clus_centroids_df()
        n_levels = clus_centroids.shape[0]
        n_radarparams = self.params.size
        lims = limitslist(np.arange(0, n_levels+1, int(n_levels/n_radarparams)))
        dfs = OrderedDict()
        for lim, param in zip(lims, self.params):
            df = clus_centroids.iloc[lim[0]:lim[1], :]
            # we don't know row names here
            df.index = pd.RangeIndex(stop=df.index.size)
            dfs[param] = df
        pn = pd.Panel(dfs)
        rw = self.radar_weight_factors
        if rw is not None:
            for field, weight_factor in rw.items():
                pn[field] = pn[field]/weight_factor
        return pn, extra

    def prepare_data(self, data, extra_df=None, n_components=0, save=True):
        """prepare data for clustering or classification"""
        data_scaled = data.copy()
        metadata = dict(fields=data_scaled.items.values,
                        hlimits=(data_scaled.minor_axis.min(),
                                 data_scaled.minor_axis.max()))
        rw = self.radar_weight_factors
        if rw is not None:
            for field, weight_factor in rw.items():
                data_scaled[field] = data_scaled[field]*weight_factor
        data_df = learn.pn2df(data_scaled)
        if self.pca is None:
            self.pca = pca_fit(data_df, n_components=n_components)
        if self.reduced:
            data = pd.DataFrame(self.pca.transform(data_df), index=data_df.index)
        else:
            data = data_df
        data.index = data.index.round('1min')
        if extra_df is not None:
            data = pd.concat([data, extra_df*self.extra_weight_factor], axis=1)
            data.dropna(inplace=True)
            data = data[~data.index.duplicated()]
        if save:
            self.data = data
            self.params = metadata['fields']
            if extra_df is not None:
                self.params_extra = pd.DataFrame(extra_df).columns.values
            self.hlimits = metadata['hlimits']
        return data
