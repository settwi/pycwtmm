from scipy.fft import fft, ifft
import numpy as np

def cwt(dat, scales, func):
    ''' Ricker CWT of a given data set and scales '''
    ret = []
    zer = np.zeros_like(dat)
    dat = np.concatenate((zer, dat, zer))

    ft = fft(dat)
    t = np.arange(dat.size)
    dt = 1. # in time steps
    for s in scales:
        pf = np.sqrt(2 * np.pi * s / dt)
        rick = pf * func(omega(s, dat.size, t) * s)
        row = ft * rick
        ret.append(ifft(row))

    # Torrence & Compo drop the imaginary part in their
    # IDL code, so I might as well do the same.
    return np.array(ret)[:, zer.size:-zer.size].real


def fourier_ricker(z):
    ''' Fourier space Ricker wavelet '''
    prefac = 2 / (np.sqrt(3 * np.sqrt(np.pi)))
    return prefac * z**2 * np.exp(-z**2 / 2)


def fourier_morlet_factory(w0: float):
    def morlet(z: np.ndarray):
        step = (z > 0).astype(int)
        prefac = np.pi**(-0.25) * step
        return prefac * np.exp(-(z - w0)**2 / 2)
    return morlet


def omega(scale, n, k):
    ''' Angular frequency as per T&C '''
    ret = (2 * np.pi * k / n) * np.ones_like(k)
    ret[k > n/2] *= -1
    return ret
