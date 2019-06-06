# coding: utf-8
import pickle
from os import path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from radcomp import learn, USER_DIR
from radcomp.vertical import preprocessing, plotting
from j24 import ensure_dir, limitslist
from j24.learn import pca_stats


META_SUFFIX = '_metadata'
MODEL_DIR = ensure_dir(path.join(USER_DIR, 'class_schemes'))

def weight_factor_str(param, value):
    out = '_' + param
    if value != 1:
        out += str(value).replace('.', '')
    return out

def scheme_name(basename='', n_eigens=30, n_clusters=20, reduced=True,
                extra_weight=0, **kws):
    if reduced:
        qualifier = '_pca'
    else:
        qualifier = ''
    if extra_weight:
        basename += weight_factor_str('t', extra_weight)
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
    #pca = decomposition.KernelPCA(kernel='poly', n_jobs=-1, degree=2,
    #                              fit_inverse_transform=True, **kws)
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


def plot_cluster_centroids(vpc, colorful_bars='blue', order=None,
                           sortby=None, n_extra_ax=0,
                           plot_counts=True, **kws):
    """class centroids pcolormesh"""
    # TODO: split massive func
    pn, extra = vpc.clus_centroids(order=order, sortby=sortby)
    order_out = pn.minor_axis
    n_extra = extra.shape[1]
    pn_plt = pn.copy() # with shifted axis, only for plotting
    pn_plt.minor_axis = pn.minor_axis-0.5
    if n_extra > 0:
        kws['n_ax_shift'] = 0#n_extra
    fig, axarr = plotting.plotpn(pn_plt, x_is_date=False,
                                 n_extra_ax=n_extra+n_extra_ax+plot_counts,
                                 has_ml=vpc.has_ml, **kws)
    if colorful_bars == True: # Might be str, so check for True.
        n_omit_coloring = 2
    else:
        n_omit_coloring = 1
    for iax in range(len(axarr)-n_omit_coloring):
        vpc.class_colors(pd.Series(pn.minor_axis), ax=axarr[iax])
    ax_last = axarr[-1]
    axn_extra = len(kws['fields']) if 'fields' in kws else pn_plt.items.size
    ax_extra = axarr[axn_extra]
    if n_extra > 0:
        extra.plot.bar(ax=ax_extra, color='black')
        ax_extra.get_legend().set_visible(False)
        ax_extra.set_ylim([-20, 1])
        ax_extra.set_ylabel(plotting.LABELS['temp_mean'])
        ax_extra.yaxis.grid(True)
    n_comp = vpc.km.n_clusters
    ax_last.set_xticks(extra.index.values)
    ax_last.set_xlim(-0.5, n_comp-0.5)
    fig = ax_last.get_figure()
    precip_type = 'rain' if vpc.has_ml else 'snow'
    axarr[0].set_title('Class centroids for {} cases'.format(precip_type))
    if colorful_bars == 'blue':
        cmkw = {}
        cmkw['cm'] = plotting.cm_blue()
    if plot_counts:
        counts = vpc.class_counts().loc[pn.minor_axis]
        plotting.plot_occurrence_counts(counts, ax=ax_last)
    if colorful_bars:
        plotting.bar_plot_colors(ax_last, pn.minor_axis,
                                 class_color_fun=vpc.class_color, **cmkw)
    fig.canvas.mpl_connect('button_press_event', vpc._on_click_plot_cl_cs)
    ax_last.set_xlabel('Class ID')
    plotting.prepend_class_xticks(ax_last, vpc.has_ml)
    return fig, axarr, order_out


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
        extra_weight (float, optional): temperature weight factor
            in clustering, 1 if unspecified
        basename (str): base name for the scheme id
    """

    def __init__(self, pca=None, km=None, hlimits=None, params=None,
                 reduced=False, n_eigens=None, n_clusters=None,
                 extra_weight=None, basename=None,
                 transformer_base=None, has_ml=False, invalid_classes=[]):
        self.pca = pca
        self.km = km  # k means
        self.hlimits = hlimits
        self.params = params
        self.params_extra = []
        self.reduced = reduced
        self.kdpmax = None
        self.data = None  # training or classification data
        self.extra_weight = extra_weight
        self.basename = basename
        self._mapping = None
        self.training_data = None # archived training data
        self.classes = None # archived training data classes
        self._n_eigens = n_eigens
        self._n_clusters = n_clusters
        self._inverse_data = None
        self._inverse_extra = None
        self._cl_ax = None
        self.invalid_classes = invalid_classes
        self.transformers = {}
        self.has_ml = has_ml
        self.setup_transform()
        self.height_index = None # training data height index

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

    @property
    def inverse_data(self):
        """lazy loading of inverse transformed data"""
        if self._inverse_data is None:
            self._inverse_data, self._inverse_extra = self.inverse_transform()
        return self._inverse_data.copy()

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

    def setup_transform(self):
        """preprocessing feature scaling transformer setup"""
        for param in self.params:
            tr = preprocessing.RadarDataScaler(param=param, has_ml=self.has_ml)
            self.transformers[param] = tr

    def feature_scaling(self, pn, inverse=False):
        """feature scaling"""
        scaled = pn.copy()
        for field, df in scaled.iteritems():
            tr = self.transformers[field]
            if inverse:
                scaled[field] = tr.inverse_transform(df.T).T
            else:
                scaled[field] = tr.fit_transform(df)
        return scaled

    def name(self):
        """scheme name string"""
        if self.basename is None:
            raise TypeError('basename is not set')
        return scheme_name(basename=self.basename, n_eigens=self._n_eigens,
                           n_clusters=self.n_clusters,
                           reduced=self.reduced,
                           extra_weight=self.extra_weight)

    def get_class_list(self):
        """class number range"""
        return range(self.n_clusters)

    def class_labels(self):
        prefix = 'r' if self.has_ml else 's'
        return [prefix + str(cl) for cl in self.get_class_list()]

    def save(self, **kws):
        """Save scheme to default location in pickle format."""
        with open(model_path(self.name(**kws)), 'wb') as f:
            pickle.dump(self, f)

    def train(self, data=None, n_eigens=None, extra_df=None, **kws):
        """Perform clustering of the training data to initialize classes."""
        self.height_index = data.minor_axis
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
        self.classes = self.prep_classes(cl_arr, training_data.index)
        self.classes.name = 'class'

    def prep_classes(self, cl_arr, index):
        """Map classes as Series"""
        self.map_components()
        classes = self._mapping[cl_arr].values
        return pd.Series(data=classes, index=index)

    def classify(self, data_scaled, **kws):
        """classify scaled observations"""
        data = self.prepare_data(data_scaled, **kws)
        cl_arr = self.km.predict(data)
        classes = self.prep_classes(cl_arr, data.index)
        tr = self.classes
        td = self.training_data
        training_classes = tr.loc[tr.index.difference(classes.index)].copy()
        training_data = td.loc[td.index.difference(data.index)].copy()
        cl_silh = pd.concat((training_classes, classes))
        data_silh = pd.concat((training_data, data))
        silh = silhouette_samples(data_silh, cl_silh)
        silh = pd.Series(data=silh, index=cl_silh.index).loc[classes.index]
        return classes, silh

    def valid_classes(self):
        valid = list(self.get_class_list())
        for i in sorted(self.invalid_classes, reverse=True):
            del valid[i]
        return valid

    def map_components(self):
        """Set class mapping, return sorted components and extra parameters"""
        centroids = self.km.cluster_centers_
        n_extra = len(self.params_extra)
        if n_extra < 1:
            components = centroids
            extra = []
        else:
            components = centroids[:, :-n_extra]
            extra = centroids[:, -n_extra:]/self.extra_weight
        # sort classes by first pca component
        components, self._mapping = sort_by_column(components, by=0)
        extra_df = pd.DataFrame(extra, columns=self.params_extra)
        if not extra_df.empty:
            extra_df = extra_df.loc[self._mapping.sort_values().index]
            extra_df.reset_index(drop=True, inplace=True)
        return components, extra_df

    def df2pn(self, data):
        """inverse transformed DataFrame to Panel"""
        n_levels = data.shape[0]
        levels_per_var = int(n_levels/self.params.size)
        lims = limitslist(np.arange(0, n_levels+1, levels_per_var))
        dfs = OrderedDict()
        for lim, param in zip(lims, self.params):
            df = data.iloc[lim[0]:lim[1], :]
            # we don't know row names here
            df.index = pd.RangeIndex(stop=df.index.size)
            dfs[param] = df
        pn = pd.Panel(dfs)
        return pn

    def inverse_transform(self, pc=None):
        """inverse transform from PCA to radar variable space"""
        pc = pc or self.data
        n_extra = len(self.params_extra)
        if n_extra < 1:
            extra = []
        else:
            pc = pc.iloc[:, :-n_extra]
            extra = pc.iloc[:, -n_extra:]/self.extra_weight
        normal = pd.DataFrame(self.pca.inverse_transform(pc).T)
        return self.df2pn(normal), extra

    def _clus_centroids_df(self):
        """
        cluster centroids DataFrame, extra parameters in separate DataFrame
        """
        components, extra_df = self.map_components()
        if self.reduced:
            centroids = self.pca.inverse_transform(components)
        cen = pd.DataFrame(centroids.T).loc[:, self.valid_classes()]
        try:
            extra_df = extra_df.loc[:, self.valid_classes()]
        except KeyError:
            pass
        return cen, extra_df

    def clus_centroids_pn(self):
        """
        cluster centroids Panel, extra parameters in separate DataFrame
        """
        clus_centroids, extra = self._clus_centroids_df()
        pn = self.df2pn(clus_centroids)
        return pn, extra

    def clus_centroids(self, order=None, sortby=None):
        """cluster centroids translated to original units"""
        clus_centroids, extra = self.clus_centroids_pn()
        clus_centroids.major_axis = self.height_index
        decoded = self.feature_scaling(clus_centroids, inverse=True)
        if (sortby is not None) and (not order):
            if isinstance(sortby, str) and extra.empty:
                order = extra.sort_values(by=sortby).index
            elif isinstance(sortby, pd.Series):
                order = sortby.sort_values().index
            else:
                raise ValueError('sortby must be series or extra column name.')
        if order is not None:
            if extra.empty:
                return decoded.loc[:, :, order], extra
            return decoded.loc[:, :, order], extra.loc[order]
        return decoded, extra

    def plot_cluster_centroids(self, **kws):
        """plot_cluster_centroids wrapper"""
        return plot_cluster_centroids(self, **kws)

    def plot_centroid(self, n, **kws):
        """Plot centroid for class n."""
        cen, t = self.clus_centroids()
        data = cen.minor_xs(n)
        axarr = plotting.plot_vps(data, has_ml=self.has_ml, **kws)
        titlestr = 'Class {} centroid'.format(n)
        if self.extra_weight:
            titlestr += ', $T_{{s}}={t:.1f}^{{\circ}}$C'.format(t=t['temp_mean'][n])
        axarr[1].set_title(titlestr)
        return axarr

    def precip_classes(self):
        """select potentially precipitating classes"""
        pn = self.clus_centroids()[0]
        zmean = pn.loc['zh'].mean()
        return zmean[zmean > -9].index

    def class_color_mapping(self):
        selection = self.precip_classes()
        mapping = pd.Series(index=selection, data=range(selection.size))
        return mapping

    def class_color(self, *args, **kws):
        """color associated to a given class number"""
        mapping = self.class_color_mapping()
        return plotting.class_color(*args, mapping=mapping, **kws)

    def class_colors(self, classes=None, **kws):
        if classes is None:
            classes = self.classes
        mapping = self.class_color_mapping()
        return plotting.class_colors(classes, mapping=mapping, **kws)

    def class_counts(self):
        """occurrences of each class"""
        count = self.classes.groupby(self.classes).count()
        count.name = 'count'
        return count

    def _on_click_plot_cl_cs(self, event):
        """click a class centroid to plot it"""
        try:
            i = int(round(event.xdata))
        except TypeError: # clicked outside axes
            return
        ax, update, axkws = plotting.handle_ax(self._cl_ax)
        ticklabels = event.inaxes.axes.get_xaxis().get_majorticklabels()
        classes = list(map(lambda la: int(la.get_text()), ticklabels))
        classn = classes[i]
        self._cl_ax = self.vpc.plot_centroid(classn, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def scatter_class_pca(self, **kws):
        """plotting.scatter_class_pca wrapper"""
        return plotting.scatter_class_pca(self.data, self.classes,
                                          color_fun=self.class_color, **kws)

    def silhouette_coef(self):
        """silhouette coefficient of each profile"""
        from sklearn.metrics import silhouette_samples
        sh_arr = silhouette_samples(self.data, self.classes)
        return pd.Series(index=self.classes.index, data=sh_arr)

    def silhouette_score(self, cols=(0, 1, 2), weights=1):
        """silhouette score"""
        if cols == 'all':
            if self.has_ml:
                weights = 1
            else:
                weights = np.ones(self.data.shape[1])
                ew = self.extra_weight
                weights[:-1] = ew
            class_data = self.data*weights
        else:
            class_data = self.data.loc[:, cols]*weights
        return silhouette_score(class_data, self.classes)

    def plot_silhouette(self, ax=None, **kws):
        """plot silhouette analysis"""
        ax = ax or plt.gca()
        s_coef = self.silhouette_coef()
        s_groups = s_coef.groupby(self.classes)
        y_lower = 10
        for cname, clust in s_groups:
            if cname not in self.precip_classes():
                continue
            color = self.class_color(cname)
            cluster = clust.sort_values()
            y_upper = y_lower + cluster.size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster,
                             facecolor=color, edgecolor=color)
            y_lower = y_upper + 30
            #ax.text(-0.05, y_lower + 0.5*cluster.size, str(cname))
        ax.axvline(x=self.silhouette_score(**kws), color="red", linestyle="--")
        ax.set_xlabel('silhouette coefficient')
        ax.set_ylabel('classes')
        ax.set_yticks([])

    def prepare_data(self, data, extra_df=None, n_components=0, save=True):
        """prepare data for clustering or classification"""
        data_scaled = data.copy()
        metadata = dict(fields=data_scaled.items.values,
                        hlimits=(data_scaled.minor_axis.min(),
                                 data_scaled.minor_axis.max()))
        data_df = learn.pn2df(data_scaled)
        if self.pca is None:
            self.pca = pca_fit(data_df, n_components=n_components)
        if self.reduced:
            data = pd.DataFrame(self.pca.transform(data_df), index=data_df.index)
        else:
            data = data_df
        data.index = data.index.round('1min')
        if extra_df is not None:
            data = pd.concat([data, extra_df*self.extra_weight], axis=1)
            data.dropna(inplace=True)
            data = data[~data.index.duplicated()]
        if save:
            self.data = data
            self.params = metadata['fields']
            if extra_df is not None:
                self.params_extra = pd.DataFrame(extra_df).columns.values
            self.hlimits = metadata['hlimits']
        return data
