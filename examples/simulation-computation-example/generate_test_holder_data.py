import cwtmm.modulus_maxima as cmm
import cwtmm.good_cwt as good
import numpy as np
import os
import pickle
from scipy import stats as st

def main():
    np.random.seed(29587)
    out_dir = 'example-confligurmirators'
    os.makedirs(out_dir, exist_ok=True)

    probs = {
        'antipersistent': 0.0,
        'brownian': 0.5,
        'persistent': 0.99,
    }

    step = 0.25
    moments = np.arange(-8, 8 + step, step)
    num_elts = 2000
    wavelet_scales = 2 ** (np.linspace(np.log2(4), np.log2(num_elts / 2), num=1000))

    for n, p in probs.items():
        print('start', n)
        sig = make_walk(p, num=num_elts, step_dist=st.uniform)
        sig = 1000 * sig / sig.max()
        t = np.arange(sig.size)
        wt = good.cwt(sig, wavelet_scales)
        confligurmirator = cmm.ContinuousWaveletTransformModulusMaximaConfligurimator(
            wt, wavelet_scales, t,
            clip_edges=5,
            raw_signal=sig
        )

        power_ratio = 2 * np.log2(np.arange(num_elts // 2) + 2)
        confligurmirator.connect_wavelet_modulus_maxima(
            allowed_power_ratio=power_ratio,
            allowed_time_differences=np.log2(wavelet_scales)
        )
        confligurmirator.compute_multifractal(
            moments, verbose=True
        )
        with open(f'{out_dir}/cwtmm-{n}.pkl', 'wb') as f:
            pickle.dump(confligurmirator, f)

        print(f'done with {n} at flip-probability {p:.2f}')


def make_walk(continue_same_direction_prob: float, num: int, step_dist=st.uniform):
    ret = np.zeros(num)
    ret[1] = step_dist.rvs()
    steps = np.abs(step_dist.rvs(size=num))
    for i in range(2, num):
        direction = np.sign(ret[i-1] - ret[i-2])
        if np.random.rand() < continue_same_direction_prob:
            ret[i] = ret[i-1] + (steps[i] * direction)/np.sqrt(num)
        else:
            ret[i] = ret[i-1] + (steps[i] * (-direction))/np.sqrt(num)
    return ret

if __name__ == '__main__': main()
