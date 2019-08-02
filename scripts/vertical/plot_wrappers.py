# coding: utf-8
"""wrappers for vpc paper 1 plots"""

from os import path

import matplotlib.pyplot as plt

from radcomp.vertical import plotting, RESULTS_DIR

from j24 import ensure_join

import conf


SAVE_DEFAULT = True
SAVE_KWS = dict(bbox_inches='tight', dpi=300)


def subdir_vpc(vpc, subdir):
    name = vpc.name()
    return ensure_join(RESULTS_DIR, subdir, name)


def plot_cluster_centroids(vpc, save=SAVE_DEFAULT, **kws):
    """plot_cluster_centroids wrapper"""
    fig, axarr, _ = vpc.plot_cluster_centroids(fig_scale_factor=0.8)
    if save:
        savedir = subdir_vpc(vpc, 'classes_summary')
        fig.savefig(path.join(savedir, 'centroids.png'), **SAVE_KWS)


def plot_silhouette():
    return


def plot_rain_case(cases_r, save=SAVE_DEFAULT):
    c = cases_r.loc['140812T02'].case
    fig, axarr = c.plot(params=['kdp', 'zh', 'zdr'],
                        n_extra_ax=0, plot_extras=['silh', 'cl'],
                        t_contour_ax_ind='all',
                        t_levels=[-40, -20, -10, -8, -3],
                        fig_scale_factor=0.75)
    formatter = plotting.concise_formatter()
    axarr[-1].xaxis.set_major_formatter(formatter)
    if save:
        fig.savefig(path.join(conf.P1_FIG_DIR, 'case_rain.png'), **SAVE_KWS)
    return fig, axarr



def plot_snow_case(cases_s, save=SAVE_DEFAULT):
    c = cases_s.loc['140215T17-16T02'].case
    fig, axarr = c.plot(params=['kdp', 'zh', 'zdr'],
                        n_extra_ax=0, plot_extras=['ts', 'silh', 'cl'],
                        t_contour_ax_ind='all',
                        t_levels=[-40, -20, -10, -8, -3],
                        fig_scale_factor=0.75)
    formatter = plotting.concise_formatter()
    axarr[-1].xaxis.set_major_formatter(formatter)
    if save:
        fig.savefig(path.join(conf.P1_FIG_DIR, 'case_snow.png'), **SAVE_KWS)
    return fig, axarr


def boxplot_t_combined(c, save=SAVE_DEFAULT, **kws):
    """boxplot_t_combined wrapper"""
    fig, ax, bp_top = plotting.boxplot_t_combined(c, i_dis=range(5), **kws)
    if save:
        savedir = subdir_vpc(c.vpc, 'classes_summary')
        fig.savefig(path.join(savedir, 'boxplot_t_combined.png'), **SAVE_KWS)


def boxplot_t_comb_both(cc_r, cc_s, **kws):
    """boxplot_t_combined for both rain and snow"""
    gs_kw = {'width_ratios': [cc_r.vpc.n_clusters, cc_s.vpc.n_clusters]}
    fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, dpi=110,
                              figsize=(10,4), gridspec_kw=gs_kw)
    plotting.boxplot_t_combined(cc_r, ax=axarr[0], **kws)
    plotting.boxplot_t_combined(cc_s, i_dis=range(5), ax=axarr[1], **kws)
    axarr[1].set_ylabel('')
    for ax in axarr:
        ax.set_xlabel('')
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('profile class')
    fig.tight_layout()
    ax.invert_yaxis()
    return fig, axarr


def conv_fmt(flag):
    """convective formatter"""
    if flag<0.5:
        return 's'
    if flag>0.5:
        return 'c'
    return ''


def datetime_fmt(t):
    """datetime formatter"""
    return t.strftime('%Y-%m-%d %H:%M')


def cases2latex(cases):
    """cases dataframe to latex table"""
    outfmt = path.join(conf.P1_FIG_DIR, 'cases_{}.tex')
    if 'convective' in cases:
        cases_out = cases.loc[:,('start','end','convective')]
        outfile = outfmt.format('r')
        formatters = (datetime_fmt, datetime_fmt, conv_fmt)
    else:
        cases_out = cases.loc[:, ('start', 'end')]
        outfile = outfmt.format('s')
        formatters = (datetime_fmt, datetime_fmt)
    cases_out.to_latex(outfile, index=False, formatters=formatters)



def plot_centroids_ensemble(cc):
    """"""
    from scr_plot_ensemble import lineboxplots
    name = conf.SCHEME_ID_RAIN if cc.has_ml else conf.SCHEME_ID_SNOW
    savedir = ensure_join(RESULTS_DIR, 'classes_summary', name, 'class_vp_ensemble')
    fields = ['kdp', 'zdr', 'zh']
    if not cc.has_ml:
        fields.append('T')
    axarrlist = lineboxplots(cc, savedir=savedir, fields=fields)
    return axarrlist



if __name__ == '__main__':
    plt.ioff()
    plt.close('all')
    #fig, ax, bp_top = plotting.boxplot_t_combined(cc, i_dis=range(5))
    #boxplot_t_combined(cc)

    #fig, _ = boxplot_t_comb_both(cc_r, cc_s)
    #fig.savefig(path.join(conf.P1_FIG_DIR, 't_tops.png'), **SAVE_KWS)

    #plot_cluster_centroids(cc_r.vpc)
    #plot_cluster_centroids(cc_s.vpc)

    #fig_r, axarr_r = plot_rain_case(cases_r)
    #fig_s, axarr_s = plot_snow_case(cases_s)

    plot_centroids_ensemble(cc_r)
    plot_centroids_ensemble(cc_s)


