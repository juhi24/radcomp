# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as mpl
#import matplotlib.pyplot as plt
import radcomp.visualization as vis
from scipy.io import loadmat
from os import path
from functools import partial
from radcomp.vertical import (filtering, classification, plotting, insitu,
                              NAN_REPLACEMENT)
from radcomp import vertical, arm, HOME, USER_DIR
from j24 import home, daterange2str

DATA_DIR = path.join(HOME, 'DATA', 'vprhi')
COL_START = 'start'
COL_END = 'end'
SCALING_LIMITS = {'ZH': (-10, 30), 'ZDR': (0, 3), 'zdr': (0, 3),
                  'KDP': (0, 0.5), 'kdp': (0, 0.15)}

def case_id_fmt(t_start, t_end=None, fmt='{year}{month}{day}', hour_fmt='%H',
                day_fmt='%d', month_fmt='%m', year_fmt='%y'):
    return daterange2str(t_start, t_end, dtformat=fmt, day_fmt=day_fmt,
                         month_fmt=month_fmt, year_fmt=year_fmt).lower()

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
    cases_list = [Case.from_dtrange(row[1][COL_START], row[1][COL_END]) for row in dts.iterrows()]
    dts['case'] = cases_list
    return dts


def dt2path(dt, datadir):
    return path.join(datadir, dt.strftime('%Y%m%d_IKA_VP_from_RHI.mat'))


def vprhimat2pn(datapath):
    """Read vertical profile mat files to Panel."""
    data = loadmat(datapath)['VP_RHI']
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
        if reverse:
            data *= SCALING_LIMITS[field][1]
            data += SCALING_LIMITS[field][0]
        else:
            data -= SCALING_LIMITS[field][0]
            data *= 1.0/SCALING_LIMITS[field][1]
        scaled[field] = data
    return scaled

def finish_cl_data_plot(axarr):
    for ax in axarr:
        param=ax.get_lines()[0].get_label().upper()
        ax.set_ylim(vis.VMINS[param], vis.VMAXS[param])
        ax.set_ylabel(vis.LABELS[param])
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(vertical.m2km))
    ax.set_xlabel('height, km')
    plotting.rotate_tick_labels(0, ax=ax)
    return axarr


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
        axkws = dict(ax=cl_ax)
        update = True
    return ax_out, update, axkws


def load_pluvio(start=None, end=None, kind='400'):
    import baecc.instruments.pluvio as pl
    name = 'pluvio{}'.format(str(kind))
    hdfpath = path.join(home(), 'DATA', 'pluvio14-16.h5')
    data = pd.read_hdf(hdfpath, key=name)[start:end]
    pluv = pl.Pluvio(data=data, name=name)
    return pluv


class Case:
    def __init__(self, data=None, cl_data=None, cl_data_scaled=None,
                 classes=None, class_scheme=None, temperature=None,
                 use_temperature=False):
        self.data = data
        self.cl_data = cl_data # non-scaled classifiable data
        self.cl_data_scaled = cl_data_scaled # scaled classifiable data
        self.classes = classes
        self.class_scheme = class_scheme
        self.temperature = temperature
        self.pluvio = None
        self._cl_ax = None
        self._dt_ax = None

    @classmethod
    def from_dtrange(cls, t0, t1):
        pn = dt2pn(t0, t1)
        return cls(data=pn)

    @classmethod
    def by_combining(cls, cases, **kws):
        t = pd.concat([c.ground_temperature() for i,c in cases.case.iteritems()])
        datas = list(cases.case.apply(lambda c: c.data)) # data of each case
        data = pd.concat(datas, axis=2)
        return cls(data=data, temperature=t, **kws)

    def name(self, **kws):
        return case_id_fmt(self.t_start(), self.t_end(), **kws)

    def t_start(self):
        return self.data.minor_axis[0]

    def t_end(self):
        return self.data.minor_axis[-1]

    def load_classification(self, name, **kws):
        self.class_scheme = classification.VPC.load(name)
        self.classify(**kws)

    def prepare_cl_data(self, save=True):
        if self.data is not None:
            cl_data = prep_data(self.data, self.class_scheme)
            if save:
                self.cl_data = cl_data
            return cl_data
        return None

    def scale_cl_data(self, save=True):
        """scaled version of classification data,
        time rounded to the nearest minute"""
        if self.cl_data is None:
            self.prepare_cl_data()
        if self.cl_data is not None:
            scaled = scale_data(self.cl_data)
            if save:
                self.cl_data_scaled = scaled
            return scaled
        return None

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
        return plotting.plot_classes(self.cl_data_scaled, self.classes)

    def plot(self, params=None, interactive=True, raw=True, n_extra_ax=0,
             plot_fr=True, **kws):
        if raw:
            data = self.data
        else:
            data = self.cl_data.transpose(0,2,1)
        if params is None:
            if self.class_scheme is not None:
                params = self.class_scheme.params
            else:
                params = ['ZH', 'zdr', 'kdp']
        plot_t = 'temp_mean' in self.class_scheme.params_extra
        plot_lwe = self.pluvio is not None
        for plot_enabled in [plot_t, plot_lwe, plot_fr]:
            if plot_enabled:
                n_extra_ax += 1
        next_free_ax = -n_extra_ax
        fig, axarr = plotting.plotpn(data, fields=params,
                                     n_extra_ax=n_extra_ax, **kws)
        if plot_lwe:
            self.plot_lwe(ax=axarr[next_free_ax])
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
        if interactive:
            for ax in axarr:
                # TODO: cursor not showing
                mpl.widgets.Cursor(ax, horizOn=False, color='red', linewidth=2)
            fig.canvas.mpl_connect('button_press_event', self._on_click_plot_dt_cs)
        for ax in axarr:
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
        return fig, axarr

    def plot_t(self, ax, tmin=-20, tmax=10):
        half_dt = self.mean_delta()/2
        t = self.ground_temperature().shift(freq=half_dt)
        plotting.plot_data(t, ax=ax)
        ax.set_ylabel(plotting.LABELS['temp_mean'])
        ax.yaxis.grid(True)
        ax.set_ylim([tmin, tmax])

    def plot_lwe(self, ax, rmax=4):
        half_dt = self.mean_delta()/2
        i = self.lwe().shift(freq=half_dt)
        plotting.plot_data(i, ax=ax, label=self.pluvio.name)
        ax.set_ylim(bottom=0, top=rmax)
        ax.set_ylabel(plotting.LABELS['intensity'])
        ax.yaxis.grid(True)
        self.set_xlim(ax)

    def plot_fr(self, ax, frmin=-0.1, frmax=1):
        half_dt = self.mean_delta()/2
        fr = self.fr().shift(freq=half_dt)
        plotting.plot_data(fr, ax=ax, label='FR')
        ax.set_ylim(bottom=frmin, top=frmax)
        ax.set_ylabel(plotting.LABELS[fr.name])
        ax.yaxis.grid(True)
        self.set_xlim(ax)

    def train(self, use_temperature, **kws):
        if use_temperature:
            extra_df = self.ground_temperature()
        else:
            extra_df = None
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        return self.class_scheme.train(data=self.cl_data_scaled,
                                       extra_df=extra_df, **kws)

    def _on_click_plot_dt_cs(self, event):
        """on click plot cross section"""
        dt = mpl.dates.num2date(event.xdata).replace(tzinfo=None)
        ax, update, axkws = handle_ax(self._dt_ax)
        self._dt_ax = self.plot_cl_data_at(dt, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def _on_click_plot_cl_cs(self, event):
        n = round(event.xdata)
        ax, update, axkws = handle_ax(self._cl_ax)
        self._cl_ax = self.plot_centroid(n, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def plot_cl_data_at(self, dt, **kws):
        data = self.cl_data
        i = data.major_axis.get_loc(dt, method='nearest')
        axarr = data.iloc[:, i, :].plot(subplots=True, **kws)
        axarr[0].set_title(str(dt))
        return finish_cl_data_plot(axarr)

    def plot_centroid(self, n, **kws):
        cen, t = self.clus_centroids()
        data = cen.minor_xs(n)
        axarr = data.plot(subplots=True, legend=False, **kws)
        titlestr = 'Class {n}, $T={t:.1f}^{{\circ}}$C'
        axarr[0].set_title(titlestr.format(n=int(n), t=t.temp_mean[n]))
        return finish_cl_data_plot(axarr)

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
        clus_centroids, extra = self.class_scheme.clus_centroids_pn()
        clus_centroids.major_axis = self.cl_data_scaled.minor_axis
        decoded = scale_data(clus_centroids, reverse=True)
        if not sortby is None and not order:
            if isinstance(sortby, str):
                order = extra.sort_values(by=sortby).index
            elif isinstance(sortby, pd.Series):
                order = sortby.sort_values().index
            else:
                raise ValueError('sortby must be series or extra column name.')
        if order is not None:
            return decoded.loc[:, :, order], extra.loc[order]
        return decoded, extra

    def plot_cluster_centroids(self, colorful_bars=False, order=None,
                               sortby='temp_mean', n_extra_ax=0, **kws):
        pn, extra = self.clus_centroids(order=order, sortby=sortby)
        order_out = pn.minor_axis
        n_extra = extra.shape[1]
        pn_plt = pn.copy() # with shifted axis, only for plotting
        pn_plt.minor_axis = pn.minor_axis-0.5
        fig, axarr = plotting.plotpn(pn_plt, x_is_date=False,
                                     n_extra_ax=n_extra+n_extra_ax+1, **kws)
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
        # plot counts
        count = self.class_counts().loc[extra.index]
        count.plot.bar(ax=ax_last)
        ax_last.set_ylabel('Occurrence')
        ax_last.yaxis.grid(True)
        if colorful_bars:
            cmkw = {}
            if colorful_bars=='blue':
                blue = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254, 1.0)
                cmkw['cm'] = mpl.colors.ListedColormap([blue]*50)
            for ax in (ax_extra, ax_last):
                pa = ax.patches
                pa = np.array(pa)[list(map(lambda p: isinstance(p, mpl.patches.Rectangle), pa))]
                for i, p in enumerate(pa):
                    p.set_color(self.class_color(extra.index[i], default=(1, 1, 1, 0), **cmkw))
        fig.canvas.mpl_connect('button_press_event', self._on_click_plot_cl_cs)
        return fig, axarr, order_out

    def class_color(self, *args, **kws):
        mapping = self.class_color_mapping()
        return plotting.class_color(*args, mapping=mapping, **kws)

    def class_colors(self, *args, **kws):
        mapping = self.class_color_mapping()
        return plotting.class_colors(*args, mapping=mapping, **kws)

    def class_color_mapping(self):
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
        if self.temperature is not None:
            return self.temperature
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        if use_arm:
            t = arm.var_in_timerange(self.t_start(), t_end, var='temp_mean')
        else:
            hdfpath = path.join(home(), 'DATA', 't_fmi_14-17.h5')
            t = pd.read_hdf(hdfpath, 'data')['TC'][self.t_start():t_end]
            t.name = 'temp_mean'
        tre = t.resample('15min', base=self.base_minute()).mean()
        if save:
            self.temperature = tre
        return tre

    def load_pluvio(self, **kws):
        self.pluvio = load_pluvio(start=self.t_start(), end=self.t_end(), **kws)

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
        fr = pd.read_hdf(hdfpath, 'data')[self.t_start():t_end]
        return self.time_weighted_mean(fr, offset_half_delta=False)

    def lwp(self):
        t_end = self.t_end()+pd.Timedelta(minutes=15)
        lwp = arm.var_in_timerange(self.t_start(), t_end, var='liq',
                                   globfmt=arm.MWR_GLOB)
        return lwp.resample('15min', base=self.base_minute()).mean()

    def class_counts(self):
        count = self.classes.groupby(self.classes).count()
        count.name = 'count'
        return count

