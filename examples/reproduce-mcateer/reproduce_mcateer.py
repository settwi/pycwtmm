import numpy as np
import matplotlib.pyplot as plt

from scipy.io import readsav

import cwtmm.modulus_maxima as cmm
import cwtmm.plotting as cpl
import cwtmm.good_cwt as gcwt


'''
Reproduce the example CWTMM curves from McAteer+2007
doi.org/10.1086/518086
'''


# Data from McAteer himself! exactly the same sa in the paper (example curves)
dat = readsav('example_lightcuves_paper.sav')
want = list(k for k in dat.keys() if 'partial' not in k)

# save for later
cwtmms = dict()

# scales = wavelet widths
# dt = allowed time difference between wavelet scales for the maxima tracing
# dp = same thing as dt but for "wavelet power"
# moments = moments as defined in McAteer paper
scales = 2**(np.linspace(1, np.log2(dat['x_anti'].size), num=250))
dt = np.log2(scales)
dp = 2 * np.log2(np.arange(scales.size) + 1)
moments = np.linspace(-8, 8, num=50)

for k in want:
    v = dat[k]
    wt = gcwt.cwt(v, scales)
    cfl = cmm.ContinuousWaveletTransformModulusMaximaConfligurimator(
        wt,
        scales,
        np.arange(v.size),
        v
    )
    cfl.connect_wavelet_modulus_maxima(dp, dt)
    cfl.compute_multifractal(moments)
    cwtmms[k] = cfl
    print('done', k)


fig, axs = plt.subplots(nrows=3, figsize=(3, 8), layout='constrained')
axs = axs.flatten()

for (k, v), ax in zip(cwtmms.items(), axs):
    cpl.plot_multifractal(
        hoelder=v.exponents_singularity_spectra['hoelder'].true,
        hausdorff=v.exponents_singularity_spectra['hausdorff'].true,
        ax=ax
    )
    ax.set(title=k, xlim=(0, 1.5), ylim=(0, 1.5))

fig.savefig('mcateer-clone.png', dpi=300)
