import copy
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

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

    norm = norm or mpl.colors.SymLogNorm(
        vmin=0,
        vmax=mag.max(),
        linthresh=1
    )
    def extend(a):
        return np.concatenate((a, [a[-1] + (a[-1] - a[-2])]))
    # note to future self: do not use imshow.
    # messes up with log-spaced bins
    plot_obj = ax.pcolormesh(extend(x), extend(y), mag, norm=norm, cmap=cm)
    return fig.colorbar(plot_obj, ax=ax)


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
    ax.fill_betweenx(y=s, x1=t[-1] - coi, x2=t[-1], **kw)


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
