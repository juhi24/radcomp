# coding: utf-8
"""tools for analyzing VPs in an individual precipitation event"""
from collections import OrderedDict
from os import path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat

from radcomp.vertical import (filtering, classification, plotting, insitu, ml,
                              deriv, NAN_REPLACEMENT)
from radcomp import arm, azs
from radcomp.tools import strftime_date_range, cloudnet
from j24 import home, daterange2str

USE_LEGACY_DATA = False

if USE_LEGACY_DATA:
    DATA_DIR = path.join(home(), 'DATA', 'vprhi')
    DATA_FILE_FMT = '%Y%m%d_IKA_VP_from_RHI.mat'
else:
    DATA_DIR = path.join(home(), 'DATA', 'vprhi2')
    DATA_FILE_FMT = '%Y%m%d_IKA_vprhi.mat'
DEFAULT_PARAMS = ['zh', 'zdr', 'kdp']


def case_id_fmt(t_start, t_end=None, dtformat='{year}{month}{day}{hour}',
                day_fmt='%d', month_fmt='%m', year_fmt='%y', hour_fmt='T%H'):
    """daterange2str wrapper for date range based IDs"""
    return daterange2str(t_start, t_end, dtformat=dtformat, hour_fmt=hour_fmt,
                         day_fmt=day_fmt, month_fmt=month_fmt,
                         year_fmt=year_fmt)


def date_us_fmt(t_start, t_end, dtformat='{day} {month} {year}', day_fmt='%d',
                month_fmt='%b', year_fmt='%Y'):
    """daterange2str wrapper for US human readable date range format"""
    return daterange2str(t_start, t_end, dtformat=dtformat, day_fmt=day_fmt,
                         month_fmt=month_fmt, year_fmt=year_fmt)


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
    try:
        return pd.Panel(data_dict, major_axis=h, minor_axis=t)
    # sometimes t does not have all values
    except ValueError as e:
        if data_dict['ZH'].shape[1] == 96:
            # manouver to set correct timestamps when data missing
            t1 = t[0] + timedelta(hours=23, minutes=45)
            midnight = t1.replace(hour=0, minute=0)
            if midnight <= t[0]:
                midnight += timedelta(hours=24)
            dt = t1-midnight
            dt_extra = timedelta(minutes=15-(dt.total_seconds()/60)%15)
            dt = dt + dt_extra
            t = pd.date_range(t[0]-dt, t1-dt, freq='15min')
            print('ObsTime missing values! Replacing with generated timestamps.')
            return pd.Panel(data_dict, major_axis=h, minor_axis=t)
        else:
            raise e


def downward_gradient(df):
    """smooth downwards gradient"""
    df_smooth = df.fillna(0).apply(filtering.savgol_series, args=(19, 2))
    dfg = df_smooth.diff()
    dfg = dfg.rolling(5, center=True).mean() # smooth gradient
    dfg[df.isnull()] = np.nan
    return -dfg


def proc_indicator(pn, var='zdrg', tlims=(-20, -10)):
    """gradient to process indicator"""
    return pn[var][(pn.T < tlims[1]) & (pn.T > tlims[0])].sum()


def kdp2phidp(kdp, dr_km):
    """Retrieve phidp from kdp."""
    kdp_filled = kdp.fillna(0)
    return 2*kdp_filled.cumsum().multiply(dr_km, axis=0)


def data_range(dt_start, dt_end):
    """read raw VP data between datetimes"""
    filepath_fmt = path.join(DATA_DIR, DATA_FILE_FMT)
    fnames = strftime_date_range(dt_start, dt_end, filepath_fmt)
    pns = map(vprhimat2pn, fnames)
    pns_out = []
    for pn in pns:
        if not pn.empty:
            pns_out.append(pn)
    return pd.concat(pns_out, axis=2, sort=True).loc[:, :, dt_start:dt_end]


def prepare_pn(pn, kdpmax=np.nan):
    """Filter data and calculate extra parameters."""
    dr = pd.Series(pn.major_axis.values, index=pn.major_axis).diff().bfill()
    dr_km = dr/1000
    pn_new = pn.copy()
    pn_new['KDP_orig'] = pn_new['KDP'].copy()
    pn_new['KDP'][pn_new['KDP'] < 0] = np.nan
    pn_new['phidp'] = kdp2phidp(pn_new['KDP'], dr_km)
    kdp = pn_new['KDP'] # a view
    # remove extreme KDP values in the panel using a view
    if USE_LEGACY_DATA:
        kdp[kdp > kdpmax] = 0
    kdp[kdp < 0] = 0
    pn_new = filtering.fltr_median(pn_new)
    pn_new = filtering.fltr_nonmet(pn_new)
    # ensure all small case keys are in place
    pn_new = filtering.create_filtered_fields_if_missing(pn_new, DEFAULT_PARAMS)
    #pn_new = filtering.fltr_ground_clutter_median(pn_new)
    pn_new['kdpg'] = downward_gradient(pn_new['kdp'])
    pn_new['zdrg'] = downward_gradient(pn_new['zdr'])
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


def prepare_data(pn, fields=DEFAULT_PARAMS, hlimits=(190, 10e3), kdpmax=None):
    """Prepare data for classification. Scaling has do be done separately."""
    data = pn[fields, hlimits[0]:hlimits[1], :].transpose(0, 2, 1)
    if kdpmax is not None:
        data['KDP'][data['KDP'] > kdpmax] = np.nan
    return fillna(data)


def prep_data(pn, vpc):
    """prepare_data wrapper"""
    return prepare_data(pn, fields=vpc.params, hlimits=vpc.hlimits, kdpmax=vpc.kdpmax)


def round_time_index(data, resolution='1min'):
    """round datetime index to a given resolution"""
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
        vpc (radcomp.vertical.VPC): classification scheme
        temperature (Series): stored temperature
        pluvio (baecc.instruments.Pluvio)
    """

    def __init__(self, data=None, cl_data=None, cl_data_scaled=None,
                 classes=None, vpc=None, has_ml=False,
                 is_convective=None):
        self.data = data
        self.cl_data = cl_data
        self.cl_data_scaled = cl_data_scaled
        self.silh_score = None
        self.vpc = vpc
        self.pluvio = None
        self.has_ml = has_ml
        self.is_convective = is_convective
        self._data_above_ml = None
        self._cl_ax = None
        self._dt_ax = None
        self.cursor = None

    @classmethod
    def from_dtrange(cls, t0, t1, **kws):
        """Create a case from data between a time range."""
        kdpmax = 0.5
        if 'has_ml' in kws:
            if kws['has_ml']:
                kdpmax = 1.3
        pn = dt2pn(t0, t1, kdpmax=kdpmax)
        return cls(data=pn, **kws)

    @classmethod
    def from_mat(cls, matfile, **kws):
        """Case object from a single mat file"""
        pn = vprhimat2pn(matfile)
        data = prepare_pn(pn)
        return cls(data=data, **kws)

    @property
    def data_above_ml(self):
        """lazy loading data above ml"""
        if not self.has_ml:
            return self.data
        if self._data_above_ml is None:
            self._data_above_ml = self.only_data_above_ml()
        return self._data_above_ml

    def only_data_above_ml(self, data=None):
        """Data above ml"""
        if data is None:
            data = self.data
        data = fillna(data)
        top = self.ml_limits()[1]
        return data.apply(ml.collapse2top, axis=(2, 1), top=top)

    def name(self, **kws):
        """date range based id"""
        return case_id_fmt(self.t_start(), self.t_end(), **kws)

    def t_start(self):
        """data start time"""
        return self.data.minor_axis[0]

    def t_end(self):
        """data end time"""
        return self.data.minor_axis[-1]

    def timestamps(self, fill_value=None, round_index=False):
        """Data timestamps as Series. Optionally filled with fill_value."""
        t = self.data.minor_axis
        data = t if fill_value is None else np.full(t.size, fill_value)
        ts = pd.Series(index=t, data=data)
        if round_index:
            return round_time_index(ts)
        return ts

    def mask(self, raw=False):
        """common data mask"""
        if raw:
            return self.data['ZH'].isnull()
        return self.data['zh'].isnull()

    def load_classification(self, name=None, **kws):
        """Load a classification scheme based on its id, and classify."""
        if name is None:
            name = self.vpc.name()
        self.vpc = classification.VPC.load(name)
        self.classify(**kws)

    def prepare_cl_data(self, save=True, force_no_crop=False):
        """Prepare unscaled classification data."""
        if self.data is not None:
            cl_data = prep_data(self.data, self.vpc)
            if self.has_ml and not force_no_crop:
                top = self.ml_limits()[1]
                collapsefun = lambda df: ml.collapse2top(df.T, top=top).T
                cl_data = cl_data.apply(collapsefun, axis=(1, 2))
                cl_data = fillna(cl_data)
                if cl_data.size == 0:
                    return None
            if save and not force_no_crop:
                self.cl_data = cl_data
            return cl_data
        return None

    def scale_cl_data(self, save=True, force_no_crop=False):
        """scaled version of classification data

        time rounded to the nearest minute
        """
        cl_data = self.prepare_cl_data(save=save, force_no_crop=force_no_crop)
        if cl_data is None:
            return None
        #scaled = scale_data(cl_data).fillna(0)
        scaled = self.vpc.feature_scaling(cl_data).fillna(0)
        if save and not force_no_crop:
            self.cl_data_scaled = scaled
        return scaled

    def ml_limits(self, interpolate=True):
        """ML top using peak detection"""
        if self.vpc is None:
            nans = self.timestamps(fill_value=np.nan)
            return nans.copy(), nans.copy()
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
        try:
            z = cl_data_scaled['zh'].T
        except KeyError:
            z = cl_data_scaled['ZH'].T
        rho = self.data['RHO'].loc[z.index]
        mli = ml.indicator(zdr, z, rho)
        if save:
            self.data['MLI'] = mli
        return mli

    def classify(self, vpc=None, save=True):
        """classify based on class_scheme"""
        if vpc is not None:
            self.vpc = vpc
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        classify_kws = {}
        if 'temp_mean' in self.vpc.params_extra:
            classify_kws['extra_df'] = self.t_surface()
        if self.cl_data_scaled is not None and self.vpc is not None:
            classes, silh = self.vpc.classify(self.cl_data_scaled, **classify_kws)
            classes.name = 'class'
            if save:
                self.silh_score = silh
            return classes, silh
        return None, None

    def inverse_transform(self):
        """inverse transformed classification data"""
        pn = self.vpc.inverse_data
        pn.major_axis = self.cl_data_scaled.minor_axis
        pn.minor_axis = self.data.minor_axis
        return self.vpc.feature_scaling(pn, inverse=True)

    def plot_classes(self):
        """plot_classes wrapper"""
        return plotting.plot_classes(self.cl_data_scaled, self.vpc.classes)

    def plot(self, params=None, interactive=True, raw=True, n_extra_ax=0,
             t_contour_ax_ind=False, above_ml_only=False, t_levels=[0],
             inverse_transformed=False,
             plot_extras=['ts', 'silh', 'cl', 'lwe'], **kws):
        """Visualize the case."""
        self.load_model_temperature()
        if not self.has_ml:
            above_ml_only = False
        if raw:
            data = self.data
        else:
            data = self.cl_data.transpose(0, 2, 1)
        if above_ml_only:
            data = self.data_above_ml if raw else self.only_data_above_ml(data)
        elif inverse_transformed:
            if self.has_ml:
                above_ml_only = True
            data = self.inverse_transform()
        if params is None:
            if self.vpc is not None:
                params = self.vpc.params
            else:
                params = DEFAULT_PARAMS
        plot_classes = ('cl' in plot_extras) and (self.vpc.classes is not None)
        plot_lwe = ('lwe' in plot_extras) and (self.pluvio is not None)
        if plot_lwe:
            plot_lwe = not self.pluvio.data.empty
        plot_azs = ('azs' in plot_extras) and (self.azs().size > 0)
        plot_fr = ('fr' in plot_extras) and (self.fr().size > 0)
        plot_t = ('ts' in plot_extras) and (self.t_surface().size > 0)
        plot_silh = ('silh' in plot_extras) and (self.vpc.classes is not None)
        plot_lr = ('lr' in plot_extras)
        n_extra_ax += plot_t + plot_lwe + plot_fr + plot_azs + plot_silh
        next_free_ax = -n_extra_ax
        cmap_override = {'LR': 'seismic', 'kdpg': 'bwr', 'zdrg': 'bwr',
                         'omega': 'seismic_r'}
        if plot_lr:
            data['LR'] = data['T'].diff()
            params = np.append(params, 'LR')
        fig, axarr = plotting.plotpn(data, fields=params,
                                     n_extra_ax=n_extra_ax, has_ml=self.has_ml,
                                     cmap_override=cmap_override, **kws)
        plotfuns = OrderedDict()
        plotfuns[self.plot_t] = plot_t
        plotfuns[self.plot_silh] = plot_silh
        plotfuns[self.plot_lwe] = plot_lwe
        plotfuns[self.plot_azs] = plot_azs
        plotfuns[self.plot_fr] = plot_fr
        for plotfun, flag in plotfuns.items():
            if flag:
                plotfun(ax=axarr[next_free_ax])
                next_free_ax += 1
        # plot temperature contours
        if t_contour_ax_ind:
            if t_contour_ax_ind == 'all':
                t_contour_ax_ind = range(len(params))
            try:
                for i in t_contour_ax_ind:
                    self.plot_growth_zones(ax=axarr[i], levels=t_levels)
            except TypeError:
                warnfmt = '{}: Could not plot temperature contours.'
                print(warnfmt.format(self.name()))
        if plot_classes:
            for iax in range(len(axarr)-1):
                self.vpc.class_colors(ax=axarr[iax])
        has_vpc = (self.vpc is not None)
        if self.has_ml and has_vpc and not above_ml_only:
            for i in range(len(params)):
                self.plot_ml(ax=axarr[i])
        self.cursor = mpl.widgets.MultiCursor(fig.canvas, axarr,
                                              color='black', horizOn=True,
                                              vertOn=True, lw=0.5)
        if interactive:
            on_click_fun = lambda event: self._on_click_plot_dt_cs(event, params=params,
                                                                   inverse_transformed=inverse_transformed,
                                                                   above_ml_only=above_ml_only)
            fig.canvas.mpl_connect('button_press_event', on_click_fun)
        for ax in axarr:
            ax.xaxis.grid(True)
            ax.yaxis.grid(True)
        self.set_xlim(ax)
        axarr[0].set_title(date_us_fmt(self.t_start(), self.t_end()))
        axarr[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%H'))
        return fig, axarr

    def plot_growth_zones(self, **kws):
        """plotting.plot_growth_zones wrapper"""
        self.load_model_temperature()
        plotting.plot_growth_zones(self.data['T'], **kws)

    def plot_ml(self, linestyle='', marker='_', ax=None):
        """Plot melting layer highlighting interpolated parts."""
        ax = ax or plt.gca()
        common_kws = dict(linestyle=linestyle, marker=marker)
        _, topi = self.ml_limits(interpolate=True)
        _, top = self.ml_limits(interpolate=False)
        ax.plot(topi.index, topi.values, color='gray', zorder=5, **common_kws)
        ax.plot(top.index, top.values, color='black', zorder=6, **common_kws)
        return ax

    def shift(self, data):
        """shift data for plotting"""
        half_dt = self.mean_delta()/2
        return data.shift(freq=half_dt)

    def plot_series(self, data, ax=None, **kws):
        """Plot time series correctly shifted."""
        ax = ax or plt.gca()
        dat = self.shift(data)
        plotting.plot_data(dat, ax=ax, **kws)
        self.set_xlim(ax)
        return ax

    def plot_t(self, ax, tmin=-20, tmax=10):
        """Plot surface temperature."""
        self.plot_series(self.t_surface(), ax=ax)
        ax.set_ylabel(plotting.LABELS['temp_mean'])
        ax.set_ylim([tmin, tmax])
        return ax

    def plot_lwe(self, ax, rmax=4):
        """plot LWE"""
        self.plot_series(self.lwe(), ax=ax, label=self.pluvio.name)
        ax.set_ylim(bottom=0, top=rmax)
        ax.set_ylabel(plotting.LABELS['intensity'])
        return ax

    def plot_fr(self, ax, frmin=-0.1, frmax=1):
        """Plot riming fraction."""
        self.plot_series(self.fr(), ax=ax, label='FR')
        ax.set_ylim(bottom=frmin, top=frmax)
        ax.set_ylabel(plotting.LABELS[self.fr().name])
        return ax

    def plot_azs(self, ax, amin=10, amax=4000):
        """Plot prefactor of Z-S relation"""
        a_zs = self.azs()
        label = plotting.LABELS[a_zs.name]
        self.plot_series(a_zs, ax=ax, label=label)
        ax.set_ylabel(plotting.LABELS[a_zs.name])
        ax.set_yscale('log')
        ax.set_ylim(bottom=amin, top=amax)
        ax.set_yticks([10, 100, 1000])
        return ax

    def plot_silh(self, ax=None):
        """Plot silhouette coefficient"""
        self.plot_series(self.silh_score, ax=ax)
        ax.set_ylabel('silhouette\ncoefficient')
        ax.set_ylim(bottom=-1, top=1)
        ax.set_yticks([-1, 0, 1])
        return ax

    def train(self, **kws):
        """Train a classification scheme with scaled classification data."""
        if self.vpc.extra_weight:
            extra_df = self.t_surface()
        else:
            extra_df = None
        if self.cl_data_scaled is None:
            self.scale_cl_data()
        return self.vpc.train(data=self.cl_data_scaled,
                                       extra_df=extra_df, **kws)

    def _on_click_plot_dt_cs(self, event, params=None, **kws):
        """on click plot profiles at a timestamp"""
        try:
            dt = plotting.num2date(event.xdata)
        except TypeError: # clicked outside axes
            return
        ax, update, axkws = plotting.handle_ax(self._dt_ax)
        axkws.update(kws)
        self._dt_ax = self.plot_data_at(dt, params=params, **axkws)
        if update:
            ax.get_figure().canvas.draw()

    def nearest_datetime(self, dt, method='nearest', **kws):
        """Round datetime to nearest data timestamp."""
        i = self.data.minor_axis.get_loc(dt, method=method, **kws)
        return self.data.minor_axis[i]

    def plot_data_at(self, dt, params=None, inverse_transformed=False,
                     above_ml_only=False, **kws):
        """Plot profiles at given timestamp."""
        data_orig = self.data_above_ml if above_ml_only else self.data
        # integer location
        i = data_orig.minor_axis.get_loc(dt, method='nearest')
        dti = data_orig.minor_axis[i]
        data = data_orig.iloc[:, :, i]
        if params is not None:
            data = data[params]
        axarr = plotting.plot_vps(data, has_ml=self.has_ml, **kws)
        if inverse_transformed:
            plotting.plot_vps(self.inverse_transform().iloc[:, :, i],
                              has_ml=self.has_ml, axarr=axarr)
        if not above_ml_only and self.has_ml:
            _, ml_top = self.ml_limits(interpolate=False)
            _, ml_top_i = self.ml_limits(interpolate=True)
            for ax in axarr:
                ax.axhline(ml_top_i.loc[dti], color='gray')
                ax.axhline(ml_top.loc[dti], color='black')
        t = data_orig.minor_axis[i]
        axarr[1].set_title(str(t))
        return axarr

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

    def t_surface(self, use_arm=False, interp_gaps=True):
        """resampled ground temperature

        Returns:
            Series: resampled temperature
        """
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
        if interp_gaps:
            tre = tre.interpolate()
        return tre

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

    def load_model_data(self, variable='temperature'):
        """Load interpolated model data."""
        self.data[variable] = cloudnet.load_as_df(self.data.major_axis,
                                                  self.data.minor_axis,
                                                  variable=variable)

    def load_model_temperature(self, overwrite=False):
        """Load interpolated model temperature if not already loaded."""
        if 'T' in self.data and not overwrite:
            return
        t = cloudnet.load_as_df(self.data.major_axis, self.data.minor_axis,
                                variable='temperature') - 273.15
        self.data['T'] = t

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

    def snd(self, var='TEMP'):
        """interpolated sounding profile data"""
        from radcomp import sounding
        ts = self.timestamps().apply(sounding.round_hours, hres=12).drop_duplicates()
        ts.index = ts.values
        if ts.iloc[0] > self.t_start():
            t0 = ts.iloc[0]-timedelta(hours=12)
            ts[t0] = t0
            ts.sort_index(inplace=True)
        if ts.iloc[-1] < self.t_end():
            t1 = ts.iloc[-1]+timedelta(hours=12)
            ts[t1] = t1
            ts.sort_index(inplace=True)
        def snd_col(x):
            try:
                return sounding.read_sounding(x, index_col='HGHT')[var]
            except pd.errors.ParserError:
                return pd.Series(index=(0, 20000), data=(np.nan, np.nan))
        a = ts.apply(snd_col)
        a.interpolate(axis=1, inplace=True)
        na = self.timestamps().apply(lambda x: np.nan)
        na = pd.DataFrame(na).reindex(a.columns, axis=1)
        t = pd.concat([na, a]).sort_index().interpolate(method='time')
        if self.vpc is not None:
            hmax = self.vpc.hlimits[1]
        else:
            hmax = self.data.major_axis.max()
        return t.loc[:, 0:hmax].drop(a.index).T

    def echotop(self):
        """echo top heights"""
        return deriv.echotop(self.data['zh'])

    def t_echotop(self, fill=True):
        """model temperature at echo top"""
        t = self.data['T']
        top = self.echotop()
        if fill:
            top.fillna(self.data.major_axis[0], inplace=True)
        d = {}
        for dt, h in top.iteritems():
            d[dt] = np.nan if np.isnan(h) else t.loc[h, dt]
        return pd.Series(d)
