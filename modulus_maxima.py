import copy
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
from scipy import optimize
import cwtmm.good_cwt as gcw

rng = np.random.default_rng()
nearest = lambda a, v: np.argmin(np.abs(a - v))


CwtmmData = namedtuple('CwtmmData', ('value', 'error', 'chisq'))
@dataclass
class CwtmmHolster:
    all: np.ndarray
    true: np.ndarray


@dataclass
class Multifractal:
    hoelder: CwtmmHolster
    hausdorff: CwtmmHolster


class Wtmmizer:
    def __init__(
        self,
        time_mids: np.ndarray,
        raw_signal: np.ndarray,
        scales: np.ndarray | None=None,
    ):
        # we want to use the ricker transform!! not a different one
        # DoG with order 2 (second derivative) is good for CWTMM (look it up!)
        self.scales = copy.deepcopy(scales) if scales is not None else default_scales(time_mids)
        self.regression_scales = copy.deepcopy(self.scales)

        cwt = gcw.cwt(raw_signal, self.scales, func=gcw.fourier_ricker)
        self.cwt_matrix = cwt

        self.time_mids = copy.deepcopy(time_mids)
        self.raw_signal = copy.deepcopy(raw_signal)

        self.maxima_paths = None
        self.multifractals = Multifractal(None, None)

    @property
    def maxima_idxs(self):
        try:
            return self._maxima_idxs
        except AttributeError:
            self._maxima_idxs = [my_max(m) for m in self.modulus]
            return self._maxima_idxs

    @property
    def modulus(self) -> np.ndarray:
        try:
            return self._modulus
        except AttributeError:
            self._modulus = np.abs(self.cwt_matrix)
            return self._modulus

    def maxima_connected(self) -> np.ndarray:
        mm = np.zeros_like(self.modulus, dtype=bool)
        for path in self.maxima_paths:
            for r, c in enumerate(path):
                mm[r, c] = True
        return mm

    @property
    def ridges_per_scale(self) -> np.ndarray:
        return self.maxima_connected().sum(axis=1)

    def connect_wavelet_modulus_maxima(
        self,
        allowed_power_ratio: float | list[float] | None=None,
        allowed_time_differences: np.ndarray | None=None,
        trim_noise: bool=True,
        clip_coi: bool=True
    ):
        # defaults (work almost every time so far)
        apr = allowed_power_ratio if allowed_power_ratio is not None\
            else 2 * (1 + np.linspace(1, np.log2(self.scales.size), num=self.scales.size))
        atd = allowed_time_differences if allowed_time_differences is not None\
            else np.log2(self.scales)

        self.maxima_paths = [[i] for i in self.maxima_idxs[0]]
        for i in range(len(self.maxima_idxs) - 1):
            dt = atd[i+1]
            self._connect_two_maxima_rows(i, apr, dt)
        if trim_noise:
            self._trim_noise()
        if clip_coi:
            self._clip_coi()

    def _clip_coi(self):
        ''' clip any maxima curves corresponding to the cone of influence
            uses: Ricker cone of influence.
        '''
        # along the time axis this delimits the cone of influence
        # as per Torrence & Compo
        max_time = self.time_mids.max()
        left_coi = np.sqrt(2) * self.scales
        right_coi = max_time - left_coi

        for mi, path in enumerate(self.maxima_paths):
            for (r, c) in enumerate(path):
                crossed_coi = (
                    self.time_mids[c] < left_coi[r] or
                    self.time_mids[c] > right_coi[r]
                )
                if crossed_coi:
                    break
            # clip to where it first crossed the COI
            self.maxima_paths[mi] = path[:r]

    def _trim_noise(self):
        ''' as states McAteer,
                Noise and artificial singularities...
                display a rapidly decreasing wavelet transform power with increasing scale.
            so: remove the noise by finding the rapidly-decreasing portions.
            cf `remove_flats2.pro`
        '''
        for path_idx, path in enumerate(self.maxima_paths):
            mm_power = [self.modulus[r, c] for (r, c) in enumerate(path)]
            for r, p in enumerate(mm_power):
                found_dropoff_point = all(p > pp for pp in mm_power[r+1:])
                if found_dropoff_point:
                    self.maxima_paths[path_idx] = path[:r+1]
                    break

    def _connect_two_maxima_rows(self, idx, allowed_power_ratio, dt):
        ''' connect modulus maxima from row i --> i+1 '''
        # note that the current maxima are the already-connected ones
        longest = max(len(x) for x in self.maxima_paths)
        ongoing_anchors = []
        for (i, p) in enumerate(self.maxima_paths):
            if len(p) != longest: continue
            ongoing_anchors.append((i, p[-1]))

        for (i, cm) in ongoing_anchors:
            best_connect = self._find_best_connection(
                idx, cm, allowed_power_ratio, dt)
            if not np.isnan(best_connect):
                self.maxima_paths[i].append(best_connect)

    def _find_best_connection(self, cur_row_idx, cur_max_idx, allowed_power_ratio, allowed_dt):
        def power_test(this_pow, next_pow):
            try: pr = allowed_power_ratio[cur_row_idx]
            except TypeError: pr = allowed_power_ratio
            return (next_pow / this_pow <= pr) and (this_pow / next_pow <= pr)

        next_modula = self.modulus[cur_row_idx+1]
        cur_power = self.modulus[cur_row_idx, cur_max_idx]
        cur_time = self.time_mids[cur_max_idx]
        low_time, high_time = cur_time - allowed_dt, cur_time + allowed_dt

        best_connect = np.nan
        best_dp_diff = np.inf

        next_maxima = self.maxima_idxs[cur_row_idx+1].copy()
        for nm_idx in next_maxima:
            next_power = next_modula[nm_idx]
            next_time = self.time_mids[nm_idx]
            within_time_frame = (low_time <= next_time <= high_time)
            if within_time_frame and power_test(next_power, cur_power):
                cur_dp = np.abs(cur_power - next_power)
                connection_crit = (
                    cur_dp < best_dp_diff
                )
                if connection_crit:
                    best_dp_diff = cur_dp
                    best_connect = nm_idx

        return best_connect

    def compute_multifractal(self, moment_values: np.ndarray, verbose: bool=False):
        # summations from eqns ~10 in McAteer
        # => shift Hölder dimension by 0.5
        HO, HD = 'hoelder', 'hausdorff'
        SLOPE_SHIFTS = {HO: -0.5, HD: 0.0}
        SUMMATIONS = {
            HO: hoelder_sum,
            HD: hausdorff_sum
        }

        ret = {
            HO: CwtmmHolster(all=list(), true=list()),
            HD: CwtmmHolster(all=list(), true=list())
        }

        # want to weight by DoF at each scale
        # => weight chi2 / dof ~ chi2 * rps.
        rps = self.ridges_per_scale

        # only regress against scales with at least one ridge
        keep_scales = (rps > 0)
        weight = rps[keep_scales]
        log_scales = np.log2(self.scales)[keep_scales]
        self.regression_scales = self.scales[keep_scales]

        mc = self.maxima_connected()[keep_scales]
        cwt_modulus = self.modulus[keep_scales]
        for q in moment_values:
            for k, shift in SLOPE_SHIFTS.items():
                vals = SUMMATIONS[k](
                    cwt_modulus, mc, q
                )
                ret[k].all.append(list(vals))

                weighted_chi2 = _wtmm_chi2_factory(log_scales, vals, weight)
                res = optimize.minimize(weighted_chi2, x0=(0, 0), method='Nelder-Mead')
                slope, _ = res.x

                ret[k].true.append(
                    CwtmmData(slope + shift, np.nan, weighted_chi2(res.x))
                )

            if verbose: print("done", q)
        self.multifractals = Multifractal(ret['hoelder'], ret['hausdorff'])


def _wtmm_chi2_factory(log_scales, observed_data, ridges_per_scale):
    CHISQ_SCALE = log_scales.size - 2
    def chi2(p):
        slope, intercept = p
        # weight the chi2 by ridges per scale
        return np.sum(
            (observed_data - intercept - log_scales*slope)**2
            * ridges_per_scale
            / CHISQ_SCALE
        )
    return chi2


def boltzmann_weight(cwt: np.ndarray, maxima_indices: np.ndarray, moment: float):
    # McAteer 2007 eq 9
    ret = []
    for i in range(cwt.shape[0]):
        row = cwt[i, maxima_indices[i]]
        ret.append(row**moment / np.sum(row**moment))
    return ret


def hoelder_sum(cwt: np.ndarray, maxima_indices: np.ndarray, moment: float):
    # McAteer 2007 eq 10
    ret = []
    w = boltzmann_weight(cwt, maxima_indices, moment)
    for i in range(cwt.shape[0]):
        ret.append(np.sum(w[i] * np.log2(cwt[i, maxima_indices[i]])))
    return ret


def hausdorff_sum(cwt: np.ndarray, maxima_indices: np.ndarray, moment: float):
    # McAteer 2007 eq 11
    ret = []
    w = boltzmann_weight(cwt, maxima_indices, moment)
    for i in range(cwt.shape[0]):
        ret.append(np.sum(w[i] * np.log2(w[i])))
    return ret


def my_max(y):
    ret = []
    sz = len(y)
    for i in range(1, sz-1):
        ret.append(y[i-1] < y[i] > y[i+1])
    # NEED THE +1 BECAUSE THE RANGE ABOVE STARTS AT 1!
    return np.where(ret)[0] + 1


def default_scales(t: np.ndarray) -> np.ndarray:
    return 2 ** np.linspace(1, np.log2(t.size/2), num=1000)
