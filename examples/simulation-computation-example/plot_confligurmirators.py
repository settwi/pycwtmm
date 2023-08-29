import cwtmm.plotting as cwtpl
import cwtmm.modulus_maxima as cmm
import matplotlib as mpl
import matplotlib.gridspec as gspc
from matplotlib import pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import sys

def main():
    plt.rcParams['font.size'] = 16
    files = sys.argv[1:]
    for fn in files:
        print('start', fn)

        fig = plt.figure(figsize=(16, 12))
        gs = gspc.GridSpec(nrows=2, ncols=2, figure=fig)
        dat = load_confligurmirator(fn)

        wt_ax = fig.add_subplot(gs[0,:])
        sig_ax = fig.add_subplot(gs[1,0])
        multifrac_ax = fig.add_subplot(gs[1, 1])

        cwtpl.plot_cwt_modulus_maxima(
            x=dat.time_mids,
            y=dat.scales,
            cwt=dat.modulus,
            ax=wt_ax,
            fig=fig,
            mm=dat.maxima_connected(),
            cmap='hot',
            mm_color='blue'
        )
        wt_ax.set(
            xlabel='Signal time (time bin)',
            ylabel='Wavelet characteristic width (time bin)',
            title='Wavelet transform + modulus maxima',
            yscale='log'
        )
        e = np.array(dat.exponents_singularity_spectra['hoelder'].true)
        cwtpl.plot_multifractal(
            e,
            np.array(dat.exponents_singularity_spectra['hausdorff'].true),
            ax=multifrac_ax,
        )
        multifrac_ax.set(
            xlabel='Hoelder dimension $h(q)$',
            ylabel='Hausdorff dimension $D(h(q))$',
            title='Multifractal spectrum',
            xlim=(-2, 2),
            ylim=(0, 1.2)
        )
        multifrac_ax.axvline(0.5)
        sig_ax.plot(dat.time_mids, dat.raw_signal)
        sig_ax.set(
            xlabel='Time (time bins)',
            ylabel='Signal (arb)',
            title=f'Original signal ({pathlib.Path(fn).stem})'
        )

        fig.tight_layout()
        fig.savefig(f'{fn.removesuffix(".xz")}.png', dpi=300)
        print('done', fn)

RetType = cmm.ContinuousWaveletTransformModulusMaximaConfligurimator
def load_confligurmirator(file: str) -> RetType:
    with open(file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    main()
