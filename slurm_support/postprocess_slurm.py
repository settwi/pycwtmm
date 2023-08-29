import lzma
import pickle
import os

import numpy as np
import cwtmm.modulus_maxima as cmm

ExtractValsIn = dict[
    str, dict[
        str, list[
            list[cmm.CwtmmData]
        ]
    ]
]
ExtractValsOut = dict[
    str, dict[
        str, np.ndarray
    ]
]

def combine_slurm_outputs(direc_name: str) -> ExtractValsIn:
    ''' combine all the .xz files in `direc_name`
        into one big dictionary!
    '''
    HO, HA = 'hoelder', 'hausdorff'
    MOMENTS_KEY = 'cwtmm_moments'
    WIDTHS_KEY = 'wavelet_widths'
    KEY_FMT = '{:.0f}-{:.0f} keV'
    ret = {'aux': (aux := dict()), 'data': (data := dict())}

    data_fnames = [
        os.path.join(direc_name, fn)
        for fn in os.listdir(direc_name)
        if fn.endswith('.xz')
    ]

    for dfn in data_fnames:
        with lzma.open(dfn, 'rb') as f:
            dat = pickle.load(f)
        aux[MOMENTS_KEY] = dat[0][MOMENTS_KEY]
        aux[WIDTHS_KEY] = dat[0][WIDTHS_KEY]

        for d in dat:
            eband_key = KEY_FMT.format(*d['energy_band'])
            ho = d['data'][HO]
            ha = d['data'][HA]
            try:
                data[eband_key][HO].append(ho)
                data[eband_key][HA].append(ha)
            except KeyError:
                data[eband_key] = {HO:  [ho], HA: [ha]}

    return ret


def values_from_perturbed_runs(data: ExtractValsIn) -> ExtractValsOut:
    ''' extract just values from a dict of combined data (defined above the function)
        return the format as defined above the function
    '''
    ret = dict()
    for eband_key, exp_dicts in data['data'].items():
        ret[eband_key] = dict()
        for exp_key, many_trials in exp_dicts.items():
            just_values = [
                [v.value for v in trial]
                for trial in many_trials
            ]
            ret[eband_key][exp_key] = np.array(just_values)

    return ret
