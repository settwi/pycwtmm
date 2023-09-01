import copy
import matplotlib as mpl
import matplotlib.axes
import matplotlib.gridspec as gspc
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from scipy import ndimage

from . import modulus_maxima as mm


def plot_cwt_modulus_maxima(
    x: np.ndarray,
    y: np.ndarray,
    cwt: np.ndarray,
    ax: mpl.axes.Axes=None,
    fig: mpl.figure.Figure=None,
    mm: np.ndarray=None,
    mm_color='black',
    smear: float=None,
    take_magnitude: bool=True,
    norm=None,
    cmap=None
):
    ax = ax or plt.gca()
    fig = fig or plt.gcf()
    mag = np.abs(cwt) if take_magnitude else cwt

    cm = plt.get_cmap(cmap or 'spring').copy()
    if mm is not None:
        smear_max = copy.deepcopy(mm).astype(float)
        smear_max = ndimage.gaussian_filter(smear_max, smear if smear is not None else 0, mode='nearest')
        mag[np.nonzero(smear_max)] = -np.inf
        cm.set_bad(mm_color)
        cm.set_under(mm_color)

    no_inf = np.nan_to_num(mag, neginf=0)
    real_min = no_inf[no_inf > 0].min()
    norm = norm or mpl.colors.SymLogNorm(
        linthresh=5000*real_min,
        vmin=real_min,
        vmax=mag.max(),
    )
    def extend(a):
        return np.concatenate((a, [a[-1] + (a[-1] - a[-2])]))
    # note to future self: do not use imshow.
    # messes up with log-spaced bins
    plot_obj = ax.pcolormesh(extend(x), extend(y), mag, norm=norm, cmap=cm)
    return fig.colorbar(plot_obj, ax=ax)


def plot_summary(wm: mm.Wtmmizer, fig: matplotlib.figure.Figure=None):
    fig = fig or plt.figure(figsize=(16, 12))

    gs = gspc.GridSpec(nrows=2, ncols=2, figure=fig)
    wt_ax = fig.add_subplot(gs[0,:])
    sig_ax = fig.add_subplot(gs[1,0])
    multifrac_ax = fig.add_subplot(gs[1, 1])


    plot_cwt_modulus_maxima(
        x=wm.time_mids,
        y=wm.scales,
        cwt=wm.modulus,
        ax=wt_ax,
        fig=fig,
        mm=wm.maxima_connected(),
        cmap='bone',
        mm_color='red'
    )
    wt_ax.set(
        xlabel='Signal time (time bin)',
        ylabel='Wavelet characteristic width (time bin)',
        title='Wavelet transform + modulus maxima',
        yscale='log'
    )
    ricker_cone_of_influence(
        wm.time_mids, wm.scales, wm.cwt_matrix,
        wt_ax, facecolor='orange', alpha=0.2,
        edgecolor='blue'
    )

    plot_multifractal(
        hoelder=wm.multifractals.hoelder.true,
        hausdorff=wm.multifractals.hausdorff.true,
        ax=multifrac_ax
    )
    multifrac_ax.set(
        xlabel='Hoelder dimension $h(q)$',
        ylabel='Hausdorff dimension $D(h(q))$',
        title='Multifractal spectrum',
        xlim=(1e-5, 1.5),
        ylim=(0, 1.5)
    )
    multifrac_ax.axvline(0.5, color='green')

    sig_ax.plot(wm.time_mids, wm.raw_signal)
    sig_ax.set(
        xlabel='Time (time bins)',
        ylabel='Signal (arb)',
        title='Original signal'
    )

    return dict(
        fig=fig,
        mf_ax=multifrac_ax,
        sig_ax=sig_ax,
        wt_ax=wt_ax
    )


def ricker_cone_of_influence(
    t: np.ndarray,
    s: np.ndarray,
    cwt: np.ndarray,
    ax: matplotlib.axes.Axes,
    **coi_kw
) -> None:
    coi = np.sqrt(2) * s
    kw = dict(
        facecolor='none', hatch='X',
        edgecolor='gray', alpha=0.8
    )
    kw.update(coi_kw or {})

    # left portion
    ax.fill_betweenx(y=s, x1=0, x2=coi, **kw)
    # right portion
    ax.fill_betweenx(y=s, x1=t[-1] - coi, x2=t[-1] + np.diff(t)[-1], **kw)


def plot_exponents_singularities(
    moments: np.ndarray,
    wavelet_scales: np.ndarray,
    all_exponents: np.ndarray,
    all_hausdorff: np.ndarray,
    axs: np.ndarray,
    selected_moments: tuple | None=None
):
    m = selected_moments if selected_moments is not None else moments
    nearest = lambda a, x: np.argmin(np.abs(a - x))
    select = np.array([nearest(moments, cm) for cm in m])
    ex = np.array(all_exponents)[select]
    mfs = np.array(all_hausdorff)[select]

    ex_ax, mf_ax = axs
    log_scale = np.log2(wavelet_scales)
    for (cur_mom, cur_ex, cur_mf) in zip(m, ex, mfs):
        # sometimes the values get clipped when fitting
        # so cut off the log_scales appropriately
        ex_ax.scatter(log_scale[:len(cur_ex)], cur_ex, label=f'q = {cur_mom:.2f}', s=8)
        mf_ax.scatter(log_scale[:len(cur_mf)], cur_mf, label=f'q = {cur_mom:.2f}', s=8)

    ex_ax.set(ylabel='Hölder exponent summation')
    mf_ax.set(ylabel='Hausdorff dimension summation')
    for ax in (ex_ax, mf_ax):
        ax.set(xlabel='$\\log_2$(wavelet scale)')
        ax.legend()


def plot_multifractal(
    hoelder: np.ndarray,
    hausdorff: np.ndarray,
    ax: mpl.axes.Axes,
    scatter_kwds=None,
    error_kwds=None,
):
    hoelder = np.array(hoelder)
    hausdorff = np.array(hausdorff)
    kwd = {'ls': 'None', 'ecolor': 'black', 'zorder': 0}
    kwd.update(error_kwds or {})
    ax.errorbar(
        x=hoelder[:,0], y=hausdorff[:,0],
        yerr=hausdorff[:,1], xerr=hoelder[:,1],
        **kwd
    )

    kwd = {'s': 8, 'color': 'red'}
    kwd.update(scatter_kwds or {})
    ax.scatter(
        x=hoelder[:,0], y=hausdorff[:,0], zorder=1,
        **kwd
    )
    ax.set(xlabel='$h(q)$', ylabel='$D(h(q))$')


def hist_with_marginal(x, y, fig, mainkw=None, xkw=None, ykw=None):
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.hist2d(x, y, **mainkw)

    ax_histx.hist(x, **xkw)
    ax_histy.hist(y, orientation='horizontal', **ykw)

    return {'main': ax, 'xhist': ax_histx, 'yhist': ax_histy}
