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
# Translate again by Samuel Garcia in Agust 2025 by mixing Charlie's python code and the new isisplot6 code
# https://github.com/magland/isosplit6
# Add also and mainly the isosplit algo in spikeinterface.
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

from __future__ import annotations

import numpy as np

import warnings
import importlib

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
    import numba
else:
    HAVE_NUMBA = False


if HAVE_NUMBA:

    ##########################
    # isocut zone

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
        best_length = length

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

    # num_bins_factor = 1
    float_0 = np.array([0.0])

    @numba.jit(nopython=True)
    def isocut(samples):  # , sample_weights=None isosplit6 not handle weight anymore
        """
        Compute a dip-test to check if 1-d samples are unimodal or not.

        This correspond to the isocut6 C++ version.

        Parameters
        ----------
        samples: np.array
            Samples input to be clustered shape (num_samples, )

        Returns
        -------
        dipscore: float
            The dipscore.
            If this dipscore<2.0 then the distribution can be considered as unimodal.
        cutpoint:
            The best cutpoint to split samples in 2 clusters in case it is not unimodal.
        """

        assert samples.ndim == 1
        N = samples.size
        assert N > 0

        # if sample_weights is None:
        #     sample_weights = np.ones(N)

        sort = np.argsort(samples)
        X = samples[sort]
        # sample_weights = sample_weights[sort]

        spacings = np.diff(X)
        mask = spacings > 0
        multiplicities = np.ones(N - 1)
        log_densities = np.ones(N - 1)
        log_densities[mask] = np.log(1.0 / spacings[mask])
        log_densities[~mask] = np.log(0.000000001)

        log_densities_unimodal_fit = up_down_isotonic_regression(log_densities, multiplicities)
        peak_ind = np.argmax(log_densities_unimodal_fit)

        log_densities_unimodal_fit_times_spacings = np.exp(log_densities_unimodal_fit) * spacings

        # difficult translation of indexing from 1-based to 0-based in
        # the following few lines. this has been checked thoroughly.
        ks_left, ks_left_ind = compute_ks5(
            multiplicities[0 : peak_ind + 1],
            # densities_unimodal_fit[0 : peak_ind + 1] * spacings[0 : peak_ind + 1],
            log_densities_unimodal_fit_times_spacings[0 : peak_ind + 1],
        )
        ks_right, ks_right_ind = compute_ks5(
            multiplicities[peak_ind:][::-1],
            # densities_unimodal_fit[peak_ind:][::-1] * spacings[peak_ind:][::-1],
            log_densities_unimodal_fit_times_spacings[peak_ind:][::-1],
        )
        ks_right_ind = spacings.size - ks_right_ind

        if ks_left > ks_right:
            critical_range = slice(0, ks_left_ind)
            dipscore = ks_left
        else:
            critical_range = slice(ks_right_ind, spacings.size)
            dipscore = ks_right

        densities_resid = log_densities[critical_range] - log_densities_unimodal_fit[critical_range]
        weights_for_downup = np.ones(densities_resid.size)
        densities_resid_fit = down_up_isotonic_regression(densities_resid, weights_for_downup)
        cutpoint_ind = critical_range.start + np.argmin(densities_resid_fit)
        cutpoint = (X[cutpoint_ind] + X[cutpoint_ind + 1]) / 2

        return dipscore, cutpoint


##########################
# isosplit zone


def isosplit(
    X,
    initial_labels=None,
    n_init=200,
    max_iterations_per_pass=500,
    min_cluster_size=10,
    isocut_threshold=2.0,
    seed=None,
):
    """
    Implementtaion in python/numba of the isosplit algorithm done by Jeremy Magland
    https://github.com/magland/isosplit6

    The algo is describe here https://arxiv.org/abs/1508.04841

    In short, the idea is to run quickly a kmeans with many centroids and then to aglomerate then iteratively
    using a dip test (using the isocut() function).

    The main benefit of this algo is that the number of cluster should be automatically guess.

    Note that this implementation in pure python/numba is 2x slower than the pure C++ one.
    Half of the run time is spent in the scipy kmeans2.
    But playing with the n_init can make it faster.

    Parameters
    ----------
    X : np.array
        Samples input to be clustered shape (num_samples, num_dim)
    n_init : int, default 200
        Initial cluster number using kmeans
    max_iterations_per_pass : int, default 500
        Number of itertions per pass.
    min_cluster_size : int, default 10
        Minimum cluster size. Too small clsuters are merged with neigbors.
    isocut_threshold : float, default 2.0
        Threhold for the merging test when exploring the cluster pairs.
        Merge is applied when : dipscore < isocut_threshold.
    seed : Int | None
        Eventually a seed for the kmeans initial step.

    Returns
    -------
    labels: np.array
        Label of the samples shape (num_smaple, ) dtype int
    """

    if initial_labels is None:

        if n_init >= X.shape[0]:
            # protect against too high n_init compared to sample size
            warnings.warn(f"isosplit : n_init {n_init} is too big compared to sample size {X.shape[0]}")
            factor = min_cluster_size * 2
            n_init = max(1, X.shape[0] // factor)
        elif n_init > (X.shape[0] // min_cluster_size):
            # protect against too high n_init compared to min_cluster_size
            warnings.warn(
                f"isosplit : n_init {n_init} is too big compared to sample size {X.shape[0]} and min_cluster_size {min_cluster_size}"
            )
            factor = min_cluster_size * 2
            n_init = max(1, X.shape[0] // factor)

        # from sklearn.cluster import KMeans, MiniBatchKMeans
        # clusterer = KMeans(n_clusters=n_init)
        # labels = clusterer.fit_predict(X)

        # scipy looks faster than scikit learn for initial labels
        from scipy.cluster.vq import kmeans2

        with warnings.catch_warnings():
            # sometimes the kmeans do not found enought cluster which should not be an issue
            warnings.simplefilter("ignore")
            _, labels = kmeans2(X, n_init, minit="points", seed=seed)

        labels = ensure_continuous_labels(labels)

    else:
        labels = ensure_continuous_labels(initial_labels)

    # Implementation note : active label here is 0-base contrary to the original code
    # importantly : the initial labels is also the indices in the centroid/covmat/comparisons_made
    # this avoid to reduce these arrays at each iteration (and this cost memory allocation)
    active_labels = np.unique(labels)
    n_cluster_init = active_labels.size
    active_labels_mask = np.ones(n_cluster_init, dtype="bool")

    centroids = np.zeros((n_cluster_init, X.shape[1]), dtype=X.dtype)
    covmats = np.zeros((n_cluster_init, X.shape[1], X.shape[1]), dtype=X.dtype)
    compute_centroids_and_covmats(X, centroids, covmats, labels, active_labels, np.ones(n_cluster_init, dtype="bool"))

    # Repeat while something has been merged in the pass
    # plus we do one final pass at the end
    final_pass = True
    # Keep a matrix of comparisons that have been made in this pass
    comparisons_made = np.zeros((n_cluster_init, n_cluster_init), dtype="bool")
    while True:  # passes
        # print()
        # print('pass')

        something_merged = False
        clusters_changed_vec_in_pass = np.zeros(n_cluster_init, dtype="bool")

        iteration_number = 0

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(ncols=2)
        # # cmap = plt.colormaps['nipy_spectral'].resampled(active_labels.size)
        # cmap = plt.colormaps['nipy_spectral'].resampled(n_init)
        # # colors = {l: cmap(i) for i, l in enumerate(active_labels)}
        # colors = {i: cmap(i) for i in range(n_init)}
        # ax = axs[0]
        # ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='nipy_spectral', s=4)
        # ax.set_title(f'n={X.shape[0]} c={active_labels.size} n_init={n_init} min_cluster_size={min_cluster_size} final_pass={final_pass}')
        # ax = axs[1]
        # for i, l in enumerate(active_labels):
        #     mask = labels == l
        #     ax.plot(X[mask, :].T, color=colors[l], alpha=0.4)
        # plt.show()

        while True:  # iterations
            iteration_number += 1
            # print('  iterations', iteration_number)

            if iteration_number > max_iterations_per_pass:
                warnings.warn("isosplit : max iterations per pass exceeded")
                break

            if active_labels.size <= 1:
                break

            if active_labels.size > 1:

                # Find the pairs to compare on this iteration
                # These will be closest pairs of active clusters that have not yet
                # been compared in this pass
                pairs = get_pairs_to_compare(centroids, comparisons_made, active_labels_mask)
                # print('pairs', len(pairs))

                if len(pairs) == 0:
                    # no pairs : break this iteration
                    break

                # Actually compare the pairs -- in principle this operation could be parallelized
                # label are updated
                clusters_changed_mask, clusters_removed_mask, total_num_label_changes = compare_pairs(
                    X, labels, pairs, centroids, covmats, min_cluster_size, isocut_threshold
                )
                # print('   ', iteration_number, 'n active', np.sum(active_labels_mask), 'changed', np.sum(clusters_changed_mask), 'n merged', np.sum(clusters_removed_mask), 'labels changed', total_num_label_changes)
                # print()

                clusters_changed_vec_in_pass |= clusters_changed_mask
                # clusters_changed_vec_in_iteration |= clusters_changed

                # Update which comparisons have been made
                for ind1, ind2 in pairs:
                    comparisons_made[ind1, ind2] = True
                    comparisons_made[ind2, ind1] = True

                # Recompute the centers for those that have changed in this iteration
                compute_centroids_and_covmats(X, centroids, covmats, labels, active_labels, clusters_changed_mask)

                if np.any(clusters_removed_mask):
                    # a merge append because one cluster disappear
                    something_merged = True

                active_labels_mask &= ~clusters_removed_mask
                active_labels = np.flatnonzero(active_labels_mask)

        # zero out the comparisons made matrix only for those that have changed in this pass
        for ind1 in active_labels:
            if clusters_changed_vec_in_pass[ind1]:
                comparisons_made[ind1, :] = False
                comparisons_made[:, ind1] = False

        # new pass or not
        if something_merged:
            final_pass = False

        if final_pass:
            # This was the final pass and nothing has merged
            # print('end')
            break

        if not something_merged:
            # If we are done, do one last pass for final redistributes
            final_pass = True

        # print('final_pass', final_pass)

    labels = ensure_continuous_labels(labels)

    return labels


def ensure_continuous_labels(labels):
    """
    This ensure [0...N[ label set
    This is important in the implementation where label is also the initial index.
    """
    label_set = np.unique(labels)
    final_labels = np.empty(labels.size, dtype=labels.dtype)
    for i, label in enumerate(label_set):
        mask = labels == label
        final_labels[mask] = i
    return final_labels


# covariance and centroid are a bit slow in pure numpy, so the formal numba version with loops  is faster
# def compute_centroids_and_covmats(X, centroids, covmats, labels, label_set, to_compute_mask):
#     for label in label_set:
#         # important note here : the label is also the index in original label set
#         i = label
#         if not to_compute_mask[i]:
#             continue
#         inds = np.flatnonzero(labels == label)
#         if inds.size > 0:
#             centroids[i, :] = np.mean(X[inds, :], axis=0)
#             if inds.size > 1:
#                 # this avoid wrning for cluster of size 1
#                 covmats[i, :, :] = np.cov(X[inds, :].T)
#         else:
#             # print('empty centroids')
#             centroids[i, :] = 0.
#             covmats[i, :, :] = 0.

if HAVE_NUMBA:

    @numba.jit(nopython=True)
    def compute_centroids_and_covmats(X, centroids, covmats, labels, label_set, to_compute_mask):
        ## manual loop with numba to be faster

        count = np.zeros(centroids.shape[0], dtype="int64")
        for i in range(centroids.shape[0]):
            if to_compute_mask[i]:
                centroids[i, :] = 0.0
                covmats[i, :, :] = 0.0

        for i in range(X.shape[0]):
            ind = labels[i]
            if to_compute_mask[ind]:
                centroids[ind, :] += X[i, :]
                count[ind] += 1

        for i in range(centroids.shape[0]):
            if to_compute_mask[i] and count[i] > 0:
                centroids[i, :] /= count[i]

        for i in range(X.shape[0]):
            ind = labels[i]
            if to_compute_mask[ind]:
                centered = X[i, :] - centroids[ind, :]
                for m1 in range(X.shape[1]):
                    for m2 in range(m1, X.shape[1]):
                        v = centered[m1] * centered[m2]
                        covmats[ind, m1, m2] += v
                        covmats[ind, m2, m1] += v

        for i in range(centroids.shape[0]):
            if to_compute_mask[i] and count[i] > 0:
                covmats[i, :, :] /= count[i]

    @numba.jit(nopython=True)
    def get_pairs_to_compare(centroids, comparisons_made, active_labels_mask):
        n = centroids.shape[0]

        dists = compute_distances(centroids, comparisons_made, active_labels_mask)
        best_inds = np.argmin(dists, axis=1)

        # already_choosen = np.zeros(n, dtype="bool")
        pairs = []
        for i1 in range(n):
            if not active_labels_mask[i1]:  # or already_choosen[i1]:
                continue
            i2 = best_inds[i1]
            if (best_inds[i2] == i1) and not (np.isinf(dists[i1, i2])) and i2 > i1:  #  and not already_choosen[i2]:
                # mutual closest
                # if already_choosen[i1] or already_choosen[i2]:
                #     print("get_pairs_to_compare() louce!! already_choosen", i1, i2)
                #     print( (i2, i1) in pairs, pairs,)
                pairs.append((i1, i2))
                # already_choosen[i1] = True
                # already_choosen[i2] = True
                dists[i1, :] = np.inf
                dists[i2, :] = np.inf
                dists[:, i1] = np.inf
                dists[:, i2] = np.inf

        return pairs

    @numba.jit(nopython=True)
    def compute_distances(centroids, comparisons_made, active_labels_mask):
        n = centroids.shape[0]
        dists = np.zeros((n, n), dtype=centroids.dtype)
        dists[:] = np.inf
        for i1 in range(n):
            if not active_labels_mask[i1]:
                continue
            for i2 in range(i1, n):
                if i1 == i2:
                    continue
                if not active_labels_mask[i2]:
                    continue
                if comparisons_made[i1, i2]:
                    continue

                d = np.sqrt(np.sum((centroids[i1, :] - centroids[i2, :]) ** 2))
                dists[i1, i2] = d
                dists[i2, i1] = d

        return dists

    @numba.jit(nopython=True)
    def merge_test(X1, X2, centroid1, centroid2, covmat1, covmat2, isocut_threshold):

        if X1.size == 0 or X2.size == 0:
            print("Error in merge test: N1 or N2 is zero. Should not be here.")
            return True, None

        V = centroid2 - centroid1
        avg_covmat = (covmat1 + covmat2) / 2.0
        inv_avg_covmat = np.linalg.inv(avg_covmat)
        V = inv_avg_covmat.astype(X1.dtype) @ V.astype(X1.dtype)
        V /= np.linalg.norm(V)

        # this two are equivalent (offset, the later is more intuitive)
        projection12 = np.concatenate((X1, X2)) @ V
        # projection12 = (np.concatenate((X1 , X2 )) - centroid1[None, :]) @ V

        dipscore, cutpoint = isocut(projection12)

        if dipscore < isocut_threshold:
            do_merge = True
            L12 = None

        else:
            do_merge = False
            L12 = (projection12 < cutpoint).astype("int32") + 1

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # count, bins = np.histogram(projection12, bins=50)
            # ax.plot(bins[:-1], count, color='k')
            # ax.axvline(cutpoint, color='m')
            # ax.set_title(f"{dipscore}")

        return do_merge, L12

    @numba.jit(nopython=True)
    def compare_pairs(X, labels, pairs, centroids, covmats, min_cluster_size, isocut_threshold):

        clusters_changed_mask = np.zeros(centroids.shape[0], dtype="bool")
        clusters_removed_mask = np.zeros(centroids.shape[0], dtype="bool")

        total_num_label_changes = 0

        for p in range(len(pairs)):
            label1, label2 = pairs[p]

            # inds1 = np.flatnonzero(labels == label1)
            # inds2 = np.flatnonzero(labels == label2)
            (inds1,) = np.nonzero(labels == label1)
            (inds2,) = np.nonzero(labels == label2)

            if (inds1.size > 0) and (inds2.size > 0):
                # if (inds1.size < min_cluster_size) and (inds2.size < min_cluster_size):
                if (inds1.size < min_cluster_size) or (inds2.size < min_cluster_size):
                    do_merge = True
                    # do_merge = False
                else:
                    X1 = X[inds1, :]
                    X2 = X[inds2, :]
                    do_merge, L12 = merge_test(
                        X1,
                        X2,
                        centroids[label1, :],
                        centroids[label2, :],
                        covmats[label1, :],
                        covmats[label2, :],
                        isocut_threshold,
                    )

                if do_merge:
                    labels[inds2] = label1
                    total_num_label_changes += inds2.size
                    clusters_changed_mask[label1] = True
                    clusters_removed_mask[label2] = True
                else:
                    # redistribute
                    something_was_redistributed = False

                    # modified_inds1 = np.flatnonzero(L12[:inds1.size] == 2)
                    # modified_inds2 = np.flatnonzero(L12[inds1.size:] == 1)
                    (modified_inds1,) = np.nonzero(L12[: inds1.size] == 2)
                    (modified_inds2,) = np.nonzero(L12[inds1.size :] == 1)

                    # protect against pure swaping between label1<>label2
                    # pure_swaping = modified_inds1.size == inds1.size and modified_inds2.size == inds2.size
                    pure_swaping = (modified_inds1.size / inds1.size + modified_inds2.size / inds2.size) >= 1.0

                    if modified_inds1.size > 0 and not pure_swaping:
                        something_was_redistributed = True
                        total_num_label_changes += modified_inds1.size
                        labels[inds1[modified_inds1]] = label2

                    if modified_inds2.size > 0 and not pure_swaping:
                        something_was_redistributed = True
                        total_num_label_changes += modified_inds2.size
                        labels[inds2[modified_inds2]] = label1

                    if something_was_redistributed:
                        clusters_changed_mask[label1] = True
                        clusters_changed_mask[label2] = True

        return clusters_changed_mask, clusters_removed_mask, total_num_label_changes
