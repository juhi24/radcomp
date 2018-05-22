# coding: utf-8
"""tools for analyzing VPs in an individual precipitation event"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os import path
from functools import partial
from radcomp.vertical import (filtering, classification, plotting, insitu, ml,
                              NAN_REPLACEMENT)
from radcomp import arm, azs, HOME, USER_DIR
from j24 import home, daterange2str

DATA_DIR = path.join(HOME, 'DATA', 'vprhi')
COL_START = 'start'
COL_END = 'end'
SCALING_LIMITS = {'ZH': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3),
                  'KDP': (0, 0.5), 'kdp': (0, 0.15)}


def case_id_fmt(t_start, t_end=None, dtformat='{year}{month}{day}{hour}',
                day_fmt='%d', month_fmt='%m', year_fmt='%y', hour_fmt='T%H'):
    """daterange2str wrapper for date range based IDs"""
    return daterange2str(t_start, t_end, dtformat=dtformat, hour_fmt=hour_fmt,
                         day_fmt=day_fmt, month_fmt=month_fmt,
                         year_fmt=year_fmt)


def date_us_fmt(t_start, t_end, dtformat='{day} {month} {year}',
                   day_fmt='%d', month_fmt='%b', year_fmt='%Y'):
    """daterange2str wrapper for US human readable date range format"""
    return daterange2str(t_start, t_end, dtformat=dtformat, day_fmt=day_fmt,
                         month_fmt=month_fmt, year_fmt=year_fmt)


def read_case_times(name):
    """Read case starting and ending times from cases directory."""
    filepath = path.join(USER_DIR, 'cases', name + '.csv')
    dts = pd.read_csv(filepath, parse_dates=[COL_START, COL_END], comment='#',
                      skip_blank_lines=True)
    indexing_func = lambda row: case_id_fmt(row[COL_START], row[COL_END])
    dts.index = dts.apply(indexing_func, axis=1)
    dts.index.name = 'id'
    return dts


def read_cases(name):
    dts = read_case_times(name)
    cases_list = []
    for cid, row in dts.iterrows():
        case_kws = dict()
        if 'ml' in row:
            case_kws['has_ml'] = bool(row['ml'])
        try:
            t_start, t_end = row[COL_START], row[COL_END]
            c = Case.from_dtrange(t_start, t_end, **case_kws)
            if c.data.empty:
                err_msg_fmt = 'No data available between {} and {}.'
                raise ValueError(err_msg_fmt.format(t_start, t_end))
            cases_list.append(c)
        except ValueError as e:
            print('Error: {}. Skipping {}'.format(e, cid))
            dts.drop(cid, inplace=True)
    dts['case'] = cases_list
    return dts


def ts_case_ids(cases):
    """case ids by timestamp"""
    cids_list = []
    for cid, c in cases.case.iteritems():
        cids_list.append(c.timestamps().apply(lambda x: cid))
    return pd.concat(cids_list)


def n_class_in_cases(class_n, cases, combined_cases=None):
    """numbers of occurrences of a given class per case in cases DataFrame"""
    case_ids = ts_case_ids(cases)
    if cases.case.iloc[0].class_scheme is not None:
        classes = pd.concat([x.classes for i, x in cases.case.iteritems()])
    elif combined_cases is not None:
        classes = combined_cases.classes
    groups = classes.groupby(case_ids)
    return groups.agg(lambda x: (x==class_n).sum())


def plot_cases_with_class(cases, class_n, **kws):
    selection = n_class_in_cases(class_n, cases).astype(bool)
    matching_cases = cases[selection]
    for name, c in matching_cases.case.iteritems():
        c.plot(**kws)


def dt2path(dt, datadir):
    return path.join(datadir, dt.strftime('%Y%m%d_IKA_VP_from_RHI.mat'))


def vprhimat2pn(datapath):
    """Read vertical profile mat files to Panel."""
    try:
        data = loadmat(datapath)['VP_RHI']
    except FileNotFoundError as e:
        print('{}. Skipping.'.format(e))
        return pd.Panel()
    fields = list(data.dtype.fields)
    fields.remove('ObsTime')
    fields.remove('height')
    str2dt = lambda tstr: pd.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S')
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
    """read raw VP data between datetimes"""
    fnames = fname_range(dt_start, dt_end)
    pns = map(vprhimat2pn, fnames)
    pns_out = []
    for pn in pns:
        if not pn.empty:
            pns_out.append(pn)
    return pd.concat(pns_out, axis=2).loc[:, :, dt_start:dt_end]


def prepare_pn(pn, kdpmax=1):
    """Filter data and calculate extra parameters."""
    dr = pd.Series(pn.major_axis.values, index=pn.major_axis).diff().bfill()
    dr_km = dr/1000
    pn_new = pn.copy()
    pn_new['KDP_orig'] = pn_new['KDP'].copy()
    pn_new['KDP'][pn_new['KDP']<0] = np.nan
    pn_new['phidp'] = kdp2phidp(pn_new['KDP'], dr_km)
    kdp = pn_new['KDP'] # a view
    # remove extreme KDP values in the panel using a view
    kdp[kdp > kdpmax] = 0
    kdp[kdp<0] = 0
    pn_new = filtering.fltr_zdr_using_rhohv(pn_new)
    pn_new = filtering.fltr_median(pn_new)
    pn_new = filtering.fltr_ground_clutter_median(pn_new)
    return pn_new


def dt2pn(dt0, dt1, **kws):
    """Read and preprocess VP data between datetimes."""
    pn_raw = data_range(dt0, dt1)
    return prepare_pn(pn_raw, **kws)


def fillna(dat, field=''):
    """Fill nan values with values representing zero scatterers."""
    data = dat.copy()
    if isinstance(data, pd.Panel):
        for field in list(data.items):
            data[field].fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    elif isinstance(data, pd.DataFrame):
        data.fillna(NAN_REPLACEMENT[field.upper()], inplace=True)
    return data


def prepare_data(pn, fields=['ZH', 'ZDR', 'kdp'], hlimits=(190, 10e3), kdpmax=None):
    """Prepare data for classification. Scaling has do be done separately."""
    data = pn[fields, hlimits[0]:hlimits[1], :].transpose(0,2,1)
    if kdpmax is not None:
        data['KDP'][data['KDP']>kdpmax] = np.nan
    return fillna(data)


def prep_data(pn, vpc):
    """prepare_data wrapper"""
    return prepare_data(pn, fields=vpc.params, hlimits=vpc.hlimits, kdpmax=vpc.kdpmax)


def scale_data(pn, reverse=False):
    """Scale radar parameters so that values are same order of magnitude."""
    scaled = pn.copy()
    for field, data in scaled.iteritems():
        try:
            if reverse:
                data *= SCALING_LIMITS[field][1]
                data += SCALING_LIMITS[field][0]
            else:
                data -= SCALING_LIMITS[field][0]
                data *= 1.0/SCALING_LIMITS[field][1]
            scaled[field] = data
        except KeyError:
            continue
    return scaled


def handle_ax(ax):
    if ax is None:
        ax_out = None
        axkws = dict()
        update = False
    else:
        cl_ax = ax
        if isinstance(cl_ax, np.ndarray):
            for ax_out in cl_ax:
                ax_out.clear()
        else:
            ax_out = cl_ax
            ax_out.clear()
        axkws = dict(axarr=cl_ax)
        update = True
    return ax_out, update, axkws


def plot_occurrence_counts(count, ax=None):
    """Bar plot occurrence counts.

    Args:
        count (Series)
    """
    if ax is None:
        ax = plt.gca()
    count.plot.bar(ax=ax)
    ax.set_ylabel('Occurrence')
    ax.yaxis.grid(True)


def round_time_index(data, resolution='1min'):
    dat = data.copy()
    ind = data.index.round(resolution)
    dat.index = ind
    return dat


class Case:
    """
    Precipitation event class for VP studies.

    Attributes:
        data (Panel)
        cl_data (Panel): non-scaled classifiable data
        cl_data_scaled (Panel): scaled classifiable data
        classes (Series): stored classification results
        class_scheme (radcomp.vertical.VPC): classification scheme
        temperature (Series): stored temperature
        pluvio (baecc.instruments.Pluvio)
    """

    def __init__(self, data=None, cl_data=None, cl_data_scaled=None,
                 classes=None, class_scheme=None, temperature=None,
                 has_ml=False):
        self.data = data
        self.cl_data = cl_data
        self.cl_data_scaled = cl_data_scaled
        self.classes = classes # is this needed?
        self.class_scheme = class_scheme
        self.temperature = temperature
        self.pluvio = None
        self.has_ml = has_ml
        self._cl_ax = None
        self._dt_ax = None

    @classmethod
    def from_dtrange(cls, t0, t1, **kws):
        """Create a case from data between a time range."""
        kdpmax = 0.5
        if 'has_ml' in kws:
            if kws['has_ml']:
                kdpmax = 1
        pn = dt2pn(t0, t1, kdpmax=kdpmax)
        return cls(data=pn, **kws)

    @classmethod
    def by_combining(cls, cases, **kws):
        """Combine a DataFrame of Case objects into one."""
        t = pd.concat([c.ground_temperature() for i,c in cases.case.iteritems()])
        datas = list(cases.case.apply(lambda c: c.data)) # data of each case
        data = pd.concat(datas, axis=2)
        return cls(data=data, temperature=t, **kws)

    def name(self, **kws):
        """date range based id"""
        return case_id_fmt(self.t_start(), self.t_end(), **kws)

    def t_start(self):
        """data start time"""
        return self.data.minor_axis[0]

    def t_end(self):
        """data end time"""
        return self.data.minor_axis[-1]

    def timestamps(self, round_index=False):
        """data timestamps as Series"""
        t = self.data.minor_axis
        ts = pd.Series(index=t, data=t)
        if round_index:
            return round_time_index(ts)
        return ts

    def load_classification(self, name, **kws):
        """Load a classification scheme based on its id, and classify."""
        self.class_scheme = classification.VPC.load(name)
        self.classify(**kws)

    def prepare_cl_data(self, save=True, force_no_crop=False):
        """Prepare unscaled classification data."""
        if self.data is not None:
            cl_data = prep_data(self.data, self.class_scheme)
            if self.has_ml and not force_no_crop:
                top = self.ml_limits()[1]
                collapsefun = lambda df: ml.collapse2top(df.T, top=top).T
                cl_data = cl_data.apply(collapsefun, axis=(1,2))
            if save and not force_no_crop:
                self.cl_data = cl_data
            return cl_data
        return None

    def scale_cl_data(self, save=True, force_no_crop=False):
        """scaled version of classification data

        time rounded to the nearest minute
        """
        cl_data = self.prepare_cl_data(save=save, force_no_crop=force_no_crop)
        scaled = scale_data(cl_data).fillna(0)
        if save and not force_no_crop:
            self.cl_data_scaled = scaled
        return scaled

    def ml_limits(self, interpolate=True):
        """ML top using peak detection"""
        if 'MLI' not in self.data:
            self.prepare_mli(save=True)
        bot, top = ml.ml_limits(self.data['MLI'], self.data['RHO'])
        if not interpolate:
            return bot, top
        return tuple(lim.interpolate().bfill().ffill() for lim in (bot, top))

    def prepare_mli(self, save=True):
        """Prepare melting layer indicator."""
        cl_data_scaled = self.scale_cl_data(force_no_crop=True)
        zdr = cl_data_scaled['zdr'].T
        z = cl_data_scaled['ZH'].T
        rho = self.data['RHO'].loc[z.index]
        mli = ml.indicator(zdr, z, rho)
        if save:
            self.data['MLI'] = mli
        return mli

    def classify(self, scheme=None, save=True):
        """classify based on class_scheme"""
        if scheme is not None:
            self.class_scheme = scheme
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        classify_kws = {}
        if 'temp_mean' in self.class_scheme.params_extra:
            classify_kws['extra_df'] = self.ground_temperature()
        if self.cl_data_scaled is not None and self.class_scheme is not None:
            classes = self.class_scheme.classify(self.cl_data_scaled, **classify_kws)
            classes.name = 'class'
            if save:
                self.classes = classes
            return classes
        return None

    def plot_classes(self):
        """plot_classes wrapper"""
        return plotting.plot_classes(self.cl_data_scaled, self.classes)

    def plot(self, params=None, interactive=True, raw=True, n_extra_ax=0,
             plot_fr=True, plot_t=True, plot_azs=True, plot_silh=True, **kws):
        """Visualize the case."""
        if raw:
            data = self.data
        else:
            data = self.cl_data.transpose(0,2,1)
        if params is None:
            if self.class_scheme is not None:
                params = self.class_scheme.params
            else:
                params = ['ZH', 'zdr', 'kdp']
        plot_lwe = self.pluvio is not None
        if plot_lwe:
            plot_lwe = not self.pluvio.data.empty
        plot_azs = plot_azs and (self.azs().size > 0)
        plot_fr = plot_fr and (self.fr().size > 0)
        plot_t = plot_t and (self.ground_temperature().size > 0)
        plot_silh = plot_silh and (self.classes is not None)
        n_extra_ax += plot_t + plot_lwe + plot_fr + plot_azs + plot_silh
        next_free_ax = -n_extra_ax
        fig, axarr = plotting.plotpn(data, fields=params,
                                     n_extra_ax=n_extra_ax, **kws)
        if plot_silh:
            self.plot_silh(ax=axarr[next_free_ax])
            next_free_ax += 1
        if plot_lwe:
            self.plot_lwe(ax=axarr[next_free_ax])
            next_free_ax += 1
        if plot_azs:
            self.plot_azs(ax=axarr[next_free_ax])
            next_free_ax += 1
        if plot_fr:
            self.plot_fr(ax=axarr[next_free_ax])
            next_free_ax += 1
        if plot_t:
            self.plot_t(ax=axarr[next_free_ax])
            next_free_ax += 1
        if self.classes is not None:
            for iax in range(len(axarr)-1):
                self.class_colors(self.classes, ax=axarr[iax])
        if self.has_ml:
            for i in range(len(params)):
                self.plot_ml(ax=axarr[i])
        if interactive:
            for ax in axarr:
                # TODO: cursor not showing
                mpl.widgets.Cursor(ax, horizOn=False, color='red', linewidth=2)
            on_click_fun = lambda event: self._on_click_plot_dt_cs(event, params=params)
            fig.canvas.mpl_connect('button_press_event', on_click_fun)
        for ax in axarr:
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
        self.set_xlim(ax)
        axarr[0].set_title(date_us_fmt(self.t_start(), self.t_end()))
        axarr[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        return fig, axarr

    def plot_ml(self, linestyle='', marker='_', ax=None):
        """Plot melting layer highlighting interpolated parts."""
        ax = ax or plt.gca()
        common_kws = dict(linestyle=linestyle, marker=marker, ax=ax)
        boti, topi = self.ml_limits(interpolate=True)
        bot, top = self.ml_limits(interpolate=False)
        topi.plot(color='gray', **common_kws)
        top.plot(color='black', **common_kws)

    def plot_t(self, ax, tmin=-20, tmax=10):
        """Plot temperature."""
        # TODO: remove copy-pasta
        half_dt = self.mean_delta()/2
        t = self.ground_temperature().shift(freq=half_dt)
        plotting.plot_data(t, ax=ax)
        ax.set_ylabel(plotting.LABELS['temp_mean'])
        ax.set_ylim([tmin, tmax])

    def plot_lwe(self, ax, rmax=4):
        """plot LWE"""
        half_dt = self.mean_delta()/2
        i = self.lwe().shift(freq=half_dt)
        plotting.plot_data(i, ax=ax, label=self.pluvio.name)
        ax.set_ylim(bottom=0, top=rmax)
        ax.set_ylabel(plotting.LABELS['intensity'])
        self.set_xlim(ax)

    def plot_fr(self, ax, frmin=-0.1, frmax=1):
        """Plot riming fraction."""
        half_dt = self.mean_delta()/2
        fr = self.fr().shift(freq=half_dt)
        plotting.plot_data(fr, ax=ax, label='FR')
        ax.set_ylim(bottom=frmin, top=frmax)
        ax.set_ylabel(plotting.LABELS[fr.name])
        self.set_xlim(ax)

    def plot_azs(self, ax, amin=10, amax=4000):
        """Plot prefactor of Z-S relation"""
        half_dt = self.mean_delta()/2
        azs = self.azs().shift(freq=half_dt)
        label = plotting.LABELS[azs.name]
        plotting.plot_data(azs, ax=ax, label=label)
        ax.set_ylabel(plotting.LABELS[azs.name])
        ax.set_yscale('log')
        ax.set_ylim(bottom=amin, top=amax)
        ax.set_yticks([10, 100, 1000])
        self.set_xlim(ax)

    def plot_silh(self, ax=None):
        """Plot silhouette coefficient"""
        ax = ax or plt.gca()
        half_dt = self.mean_delta()/2
        silh = self.silhouette_coef().shift(freq=half_dt)
        plotting.plot_data(silh, ax=ax)
        ax.set_ylabel('silhouette coefficient')
        ax.set_ylim(bottom=-1, top=1)
        ax.set_yticks([-1, 0, 1])
        self.set_xlim(ax)

    def train(self, **kws):
        """Train a classification scheme with scaled classification data."""
        if self.class_scheme.use_temperature:
            extra_df = self.ground_temperature()
        else:
            extra_df = None
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        return self.class_scheme.train(data=self.cl_data_scaled,
                                       extra_df=extra_df, **kws)

    def _on_click_plot_dt_cs(self, event, params=None):
        """on click plot profiles at a timestamp"""
        try:
            dt = mpl.dates.num2date(event.xdata).replace(tzinfo=None)
        except TypeError: # clicked outside axes
            return
        ax, update, axkws = handle_ax(self._dt_ax)
        self._dt_ax = self.plot_data_at(dt, params=params, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def _on_click_plot_cl_cs(self, event):
        """click a class centroid to plot it"""
        try:
            i = int(round(event.xdata))
        except TypeError: # clicked outside axes
            return
        ax, update, axkws = handle_ax(self._cl_ax)
        ticklabels = event.inaxes.axes.get_xaxis().get_majorticklabels()
        classes = list(map(lambda la: int(la.get_text()), ticklabels))
        classn = classes[i]
        self._cl_ax = self.plot_centroid(classn, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def plot_data_at(self, dt, params=None, **kws):
        """Plot profiles at given timestamp."""
        data_orig = self.data
        i = data_orig.minor_axis.get_loc(dt, method='nearest')
        data = data_orig.iloc[:, :, i]
        if params:
            data = data[params]
        axarr = plotting.plot_vps(data, **kws)
        t = data_orig.minor_axis[i]
        axarr[1].set_title(str(t))
        return axarr

    def plot_centroid(self, n, **kws):
        """Plot centroid for class n."""
        # TODO move to VPC
        cen, t = self.clus_centroids()
        data = cen.minor_xs(n)
        axarr = plotting.plot_vps(data, **kws)
        titlestr = 'Class {}'.format(n)
        if self.class_scheme.use_temperature:
            titlestr += ', $T={t:.1f}^{{\circ}}$C'.format(t['temp_mean'][n])
        axarr[1].set_title(titlestr)
        return axarr

    def pcolor_classes(self, **kws):
        groups = self.cl_data.groupby(self.classes)
        axarrs = groups.apply(plotting.pcolor_class, **kws)
        for i, axarr in axarrs.iteritems():
            axarr[0].set_title('Class {}'.format(i))
        figs = axarrs.apply(lambda axarr: axarr[0].get_figure())
        out = pd.concat([figs, axarrs], axis=1)
        out.columns = ['fig', 'axarr']
        return out

    def clus_centroids(self, order=None, sortby=None):
        """cluster centroids translated to original units"""
        # TODO: move to VPC
        clus_centroids, extra = self.class_scheme.clus_centroids_pn()
        clus_centroids.major_axis = self.cl_data_scaled.minor_axis
        decoded = scale_data(clus_centroids, reverse=True)
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

    def plot_cluster_centroids(self, colorful_bars=False, order=None,
                               sortby='temp_mean', n_extra_ax=0,
                               plot_counts=True, **kws):
        """class centroids pcolormesh"""
        # TODO: split massive func
        # TODO: move to VPC
        pn, extra = self.clus_centroids(order=order, sortby=sortby)
        order_out = pn.minor_axis
        n_extra = extra.shape[1]
        pn_plt = pn.copy() # with shifted axis, only for plotting
        pn_plt.minor_axis = pn.minor_axis-0.5
        fig, axarr = plotting.plotpn(pn_plt, x_is_date=False,
                                     n_extra_ax=n_extra+n_extra_ax+plot_counts,
                                     **kws)
        if colorful_bars==True: # Might be str, so check for True.
            n_omit_coloring = 2
        else:
            n_omit_coloring = 1
        for iax in range(len(axarr)-n_omit_coloring):
            self.class_colors(pd.Series(pn.minor_axis), ax=axarr[iax])
        ax_last = axarr[-1]
        ax_extra = axarr[-2]
        if n_extra>0:
            extra.plot.bar(ax=ax_extra)
            ax_extra.get_legend().set_visible(False)
            ax_extra.set_ylim([-15, 6])
            ax_extra.set_ylabel(plotting.LABELS['temp_mean'])
            ax_extra.yaxis.grid(True)
        n_comp = self.class_scheme.km.n_clusters
        ax_last.set_xticks(extra.index.values)
        ax_last.set_xlim(-0.5,n_comp-0.5)
        ax_last.set_xlabel('Class ID')
        fig = ax_last.get_figure()
        axarr[0].set_title('Class centroids')
        if plot_counts:
            plot_occurrence_counts(self.class_counts().loc[pn.minor_axis], ax=ax_last)
        if colorful_bars:
            cmkw = {}
            if colorful_bars=='blue':
                blue = (0.29803921568627451, 0.44705882352941179,
                        0.69019607843137254, 1.0)
                cmkw['cm'] = mpl.colors.ListedColormap([blue]*50)
            for ax in [ax_last]:
                pa = ax.patches
                check_rect = lambda p: isinstance(p, mpl.patches.Rectangle)
                pa = np.array(pa)[list(map(check_rect, pa))]
                for i, p in enumerate(pa):
                    p.set_color(self.class_color(pn.minor_axis[i], default=(1, 1, 1, 0), **cmkw))
        fig.canvas.mpl_connect('button_press_event', self._on_click_plot_cl_cs)
        return fig, axarr, order_out

    def scatter_class_pca(self, **kws):
        """plotting.scatter_class_pca wrapper"""
        return plotting.scatter_class_pca(self.class_scheme.data,
                                          self.classes.values,
                                          color_fun=self.class_color, **kws)

    def class_color(self, *args, **kws):
        """color associated to a given class number"""
        # TODO: move to VPC
        mapping = self.class_color_mapping()
        return plotting.class_color(*args, mapping=mapping, **kws)

    def class_colors(self, *args, **kws):
        # TODO: move to VPC
        mapping = self.class_color_mapping()
        return plotting.class_colors(*args, mapping=mapping, **kws)

    def class_color_mapping(self):
        # TODO: move to VPC
        pn = self.clus_centroids()[0]
        zmean = pn.loc['ZH'].mean()
        selection = zmean[zmean>-9].index
        mapping = pd.Series(index=selection, data=range(selection.size))
        return mapping

    def set_xlim(self, ax):
        start = self.t_start()-self.mean_delta()/2
        end = self.t_end()+self.mean_delta()/2
        ax.set_xlim(left=start, right=end)
        return ax

    def mean_delta(self):
        return plotting.mean_delta(self.data.minor_axis).round('1min')

    def base_minute(self):
        """positive offset in minutes for profile measurements after each hour
        """
        return self.data.minor_axis[0].round('1min').minute%15

    def base_middle(self):
        dt_minutes = round(self.mean_delta().total_seconds()/60)
        return self.base_minute()-dt_minutes/2

    def time_weighted_mean(self, data, offset_half_delta=True):
        dt = self.mean_delta()
        if offset_half_delta:
            base = self.base_middle()
            offset = dt/2
        else:
            base = self.base_minute()
            offset = 0
        return insitu.time_weighted_mean(data, rule=dt, base=base, offset=offset)

    def ground_temperature(self, save=False, use_arm=False):
        """resampled ground temperature

        Returns:
            Series: resampled temperature
        """
        if self.temperature is not None:
            return self.temperature
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        if use_arm:
            t = arm.var_in_timerange(self.t_start(), t_end, var='temp_mean')
        else:
            hdfpath = path.join(home(), 'DATA', 't_fmi_14-17.h5')
            if not path.exists(hdfpath):
                return pd.Series()
            t = pd.read_hdf(hdfpath, 'data')['TC'][self.t_start():t_end]
            t.name = 'temp_mean'
        tre = t.resample('15min', base=self.base_minute()).mean()
        if save:
            self.temperature = tre
        return tre

    def cl(self):
        """Classes rounded to 1min"""
        return round_time_index(self.classes)

    def azs(self, **kws):
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        data = azs.load_series()[self.t_start(): t_end]
        if data.empty:
            return pd.Series()
        return data.resample('15min', base=self.base_minute()).mean()

    def load_pluvio(self, **kws):
        """load_pluvio wrapper"""
        self.pluvio = insitu.load_pluvio(start=self.t_start(),
                                         end=self.t_end(), **kws)

    def lwe(self):
        """liquid water equivalent precipitation rate"""
        if self.pluvio is None:
            self.load_pluvio()
        i = self.pluvio.intensity()
        return self.time_weighted_mean(i, offset_half_delta=False)

    def fr(self):
        """rime mass fraction"""
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        hdfpath = path.join(home(), 'DATA', 'FR_haoran.h5')
        if not path.exists(hdfpath):
            return pd.Series()
        fr = pd.read_hdf(hdfpath, 'data')[self.t_start():t_end]
        return self.time_weighted_mean(fr, offset_half_delta=False)

    def lwp(self):
        """liquid water path"""
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        lwp = arm.var_in_timerange(self.t_start(), t_end, var='liq',
                                   globfmt=arm.MWR_GLOB)
        return lwp.resample('15min', base=self.base_minute()).mean()

    def class_counts(self):
        """occurrences of each class"""
        count = self.classes.groupby(self.classes).count()
        count.name = 'count'
        return count

    def silhouette_coef(self):
        """silhouette coefficient of each profile"""
        from sklearn.metrics import silhouette_samples
        sh_arr = silhouette_samples(self.class_scheme.data, self.classes)
        return pd.Series(index=self.classes.index, data=sh_arr)

