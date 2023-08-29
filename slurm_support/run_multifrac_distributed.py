import argparse
import json
import lzma
import multiprocessing as mp
import pickle
import sys

import numpy as np
from scipy import signal

import cwtmm.modulus_maxima as cmm

MOMENTS = np.arange(-8, 8.5, 0.5)
NUM_WIDTHS = 1000
MIN_WIDTH = 2
POWER_RATIO = 2


def main():
    args = parse_args(sys.argv[1:])

    job_name = args.job_name
    num_threads = int(args.num_threads)
    out_dir = args.out_dir
    files = args.json_files

    with mp.Pool(num_threads) as p:
        ret = p.map(single_band_multifrac, files)

    with lzma.open(f'{out_dir}/{job_name}.xz', 'w') as f:
        pickle.dump(ret, f)


def parse_args(argz: list[str]) -> argparse.Namespace:
    pr = argparse.ArgumentParser()
    pr.add_argument('--job_name')
    pr.add_argument('--num_threads')
    pr.add_argument('--out_dir')
    pr.add_argument('--json_files', nargs='*')
    return pr.parse_args(argz)


def single_band_multifrac(file_name: str) -> dict[str, object]:
    with open(file_name, 'r') as f:
        dat = json.loads(f.read())
        eb = np.array(dat['energy_band'])
        ct = np.array(dat['counts'])
        t = np.arange(ct.size)

        widths = np.logspace(
            np.log10(MIN_WIDTH),
            np.log10(ct.size / 2),
            num=NUM_WIDTHS
        )
        cwt = signal.cwt(ct, signal.ricker, widths)

        cflg = cmm.ContinuousWaveletTransformModulusMaximaConfligurimator(
            cwt_matrix=cwt,
            scales=widths,
            time_mids=t
        )

        allow_dt = np.log2(widths)
        cflg.connect_wavelet_modulus_maxima(POWER_RATIO, allow_dt)
        cflg.compute_exponents_multifractal(
            moment_values=MOMENTS,
            verbose=False
        )

    ret = dict()
    ret['data'] = {
        k: v.true for (k, v) in
        cflg.exponents_singularity_spectra.items()
    }
    ret['energy_band'] = eb
    ret['cwtmm_moments'] = MOMENTS
    ret['wavelet_widths'] = widths
    print('done', eb)
    return ret


if __name__ == '__main__':
    main()
