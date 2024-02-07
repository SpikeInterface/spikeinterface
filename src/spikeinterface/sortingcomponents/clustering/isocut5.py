from __future__ import annotations

import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


#
# Copyright 2016-2017 Flatiron Institute, Simons Foundation
# Original algorithm by Jeremy Magland and Alex Barnett,
# https://arxiv.org/abs/1508.04841
#
# Translated by Charlie Windolf from J. Magland's MEX/C++ code in June 2021
# and June 2022.
#
# J. Magland's original MATLAB/MEX available at:
# github.com/flatironinstitute/isosplit5/blob/master/matlab/jisotonic5_mex.cpp
# Python (C++/PyBind) implementation: https://github.com/magland/isosplit5_python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


if HAVE_NUMBA:

    @numba.jit(nopython=True)
    def jisotonic5(x, weights):
        N = x.shape[0]

        MSE = np.zeros(N, dtype=x.dtype)
        y = np.zeros(N, dtype=x.dtype)

        unweightedcount = np.zeros(N, dtype=np.int_)
        count = np.zeros(N, dtype=np.double)
        wsum = np.zeros(N, dtype=np.double)
        wsumsqr = np.zeros(N, dtype=np.double)
        last_index = 0

        unweightedcount[last_index] = 1
        count[last_index] = weights[0]
        wsum[last_index] = x[0] * weights[0]
        wsumsqr[last_index] = x[0] * x[0] * weights[0]

        prevMSE = 0.0
        newMSE = 0.0

        for j in range(1, N):
            last_index += 1
            unweightedcount[last_index] = 1
            count[last_index] = weights[j]
            wsum[last_index] = x[j] * weights[j]
            wsumsqr[last_index] = x[j] * x[j] * weights[j]
            MSE[j] = MSE[j - 1]

            while True:
                if last_index <= 0:
                    break
                if wsum[last_index - 1] / count[last_index - 1] < wsum[last_index] / count[last_index]:
                    break

                prevMSE = wsumsqr[last_index - 1] - wsum[last_index - 1] * wsum[last_index - 1] / count[last_index - 1]
                prevMSE += wsumsqr[last_index] - wsum[last_index] * wsum[last_index] / count[last_index]
                unweightedcount[last_index - 1] += unweightedcount[last_index]
                count[last_index - 1] += count[last_index]
                wsum[last_index - 1] += wsum[last_index]
                wsumsqr[last_index - 1] += wsumsqr[last_index]
                newMSE = wsumsqr[last_index - 1] - wsum[last_index - 1] * wsum[last_index - 1] / count[last_index - 1]
                MSE[j] += newMSE - prevMSE
                last_index -= 1

        ii = 0
        for k in range(last_index + 1):
            for cc in range(unweightedcount[k]):
                y[ii + cc] = wsum[k] / count[k]
            ii += unweightedcount[k]

        return y, MSE

    @numba.jit(nopython=True)
    def updown_arange(num_bins, dtype=np.int_):
        num_bins_1 = int(np.ceil(num_bins / 2))
        num_bins_2 = num_bins - num_bins_1
        return np.concatenate(
            (
                np.arange(dtype(1), dtype(num_bins_1 + 1)),
                np.arange(dtype(num_bins_2), dtype(1 - 1), step=-1),
            )
        )

    @numba.jit(nopython=True)
    def compute_ks4(counts1, counts2):
        c1s = counts1.sum()
        c2s = counts2.sum()
        s1 = np.cumsum(counts1)
        s1 /= c1s
        s2 = np.cumsum(counts2)
        s2 /= c2s
        ks = np.abs(s1 - s2).max()
        ks *= np.sqrt((c1s + c2s) / 2)
        return ks

    @numba.jit(nopython=True)
    def compute_ks5(counts1, counts2):
        best_ks = -np.inf
        length = counts1.size

        while length >= 4 or length == counts1.size:
            ks = compute_ks4(counts1[0:length], counts2[0:length])
            if ks > best_ks:
                best_ks = ks
                best_length = length
            length //= 2

        return best_ks, best_length

    @numba.jit(nopython=True)
    def up_down_isotonic_regression(x, weights=None):
        # determine switch point
        _, mse1 = jisotonic5(x, weights)
        _, mse2r = jisotonic5(x[::-1].copy(), weights[::-1].copy())
        mse0 = mse1 + mse2r[::-1]
        best_ind = mse0.argmin()

        # regressions. note the negatives for decreasing.
        y1, _ = jisotonic5(x[:best_ind], weights[:best_ind])
        y2, _ = jisotonic5(-x[best_ind:], weights[best_ind:])
        y2 = -y2

        return np.hstack((y1, y2))

    @numba.jit(nopython=True)
    def down_up_isotonic_regression(x, weights=None):
        return -up_down_isotonic_regression(-x, weights=weights)

    num_bins_factor = 1
    float_0 = np.array([0.0])

    @numba.jit(nopython=True)
    def isocut5(samples, sample_weights=None):
        assert samples.ndim == 1
        N = samples.size
        assert N > 0
        num_bins = int(np.ceil(np.sqrt(N / 2) * num_bins_factor))

        if sample_weights is None:
            sample_weights = np.ones(N)

        sort = np.argsort(samples)
        X = samples[sort]
        sample_weights = sample_weights[sort]

        while True:
            intervals = updown_arange(num_bins, dtype=float)
            alpha = (N - 1) / intervals.sum()
            intervals *= alpha
            # this line is the only one to translate to 0-based
            inds = np.floor(np.hstack((float_0, np.cumsum(intervals)))).astype(np.int_)
            if intervals.min() >= 1:
                break
            else:
                num_bins -= 1

        cumsum_sample_weights = np.cumsum(sample_weights)
        X_sub = X[inds]
        spacings = np.diff(X_sub)
        multiplicities = np.diff(cumsum_sample_weights[inds])
        densities = multiplicities / spacings

        densities_unimodal_fit = up_down_isotonic_regression(densities, multiplicities)
        peak_ind = np.argmax(densities_unimodal_fit)

        # difficult translation of indexing from 1-based to 0-based in
        # the following few lines. this has been checked thoroughly.
        ks_left, ks_left_ind = compute_ks5(
            multiplicities[0 : peak_ind + 1],
            densities_unimodal_fit[0 : peak_ind + 1] * spacings[0 : peak_ind + 1],
        )
        ks_right, ks_right_ind = compute_ks5(
            multiplicities[peak_ind:][::-1],
            densities_unimodal_fit[peak_ind:][::-1] * spacings[peak_ind:][::-1],
        )
        ks_right_ind = spacings.size - ks_right_ind

        if ks_left > ks_right:
            critical_range = slice(ks_left_ind)
            dipscore = ks_left
        else:
            critical_range = slice(ks_right_ind, spacings.size)
            dipscore = ks_right

        densities_resid = densities[critical_range] - densities_unimodal_fit[critical_range]
        densities_resid_fit = down_up_isotonic_regression(densities_resid, spacings[critical_range])
        cutpoint_ind = critical_range.start + np.argmin(densities_resid_fit)
        cutpoint = (X_sub[cutpoint_ind] + X_sub[cutpoint_ind + 1]) / 2

        return dipscore, cutpoint
