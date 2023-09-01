import numpy as np
import os
import pickle
from scipy import stats as st

import pycwtmm.fractional_brownian_motion as fbm
import pycwtmm.modulus_maxima as cmm

def main():
    np.random.seed(29587)
    out_dir = 'example-wms'
    os.makedirs(out_dir, exist_ok=True)

    probs = {
        'antipersistent': 0.05,
        'brownian': 0.5,
        'persistent': 0.95,
    }

    moments = np.linspace(-8, 8, num=20)
    num_elts = 1000

    for n, p in probs.items():
        print('start', n)
        sig = fbm.make_timeseries(num_elts, p)
        t = np.arange(sig.size)
        wm = cmm.Wtmmizer(t, sig)

        wm.connect_wavelet_modulus_maxima()
        wm.compute_multifractal(
            moments, verbose=True
        )
        with open(f'{out_dir}/cwtmm-{n}.pkl', 'wb') as f:
            pickle.dump(wm, f)

        print(f'done with {n} at flip-probability {p:.2f}')

if __name__ == '__main__': main()
