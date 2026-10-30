"""Numba kernel for ``AmplitudeScalingNode``, giving the same scalings as its numpy path up to rounding.

The node calls ``compute_scalings`` when numba is installed and every template is zero outside its sparsity.
"""

import numpy as np
from numba import njit

# A spike that collides with nothing gets the linregress slope of its waveform on its template, as from
# _ordinary_scaling_slope. A spike that collides, as find_collisions defines it, gets its coefficient in
# fit_collision's fit: a non-negative least-squares fit of the waveform on the colliding spikes' templates, plus an
# intercept.
#
# fit_collision builds a design matrix X with one column per spike and one row per sample and channel of the
# window, mostly zeros, and passes it to scipy.optimize.nnls. Least squares depends on X only through the normal
# equations X'X b = X'y. The Gram matrix X'X holds the dot product of each pair of templates over the samples and
# channels they share, and X'y holds each template's dot product with the waveform. This module computes those
# k x k and k terms for k spikes without building X, and solves the fit on them with Lawson-Hanson, the active-set
# method nnls also uses.
#
# cache=True keeps the compiled code on disk, so it compiles on first use rather than in every session.


@njit(cache=True)
def _is_well_conditioned(G):
    """Whether every coefficient of the fit on ``G`` can be solved from the normal equations to float32 precision."""
    # Solving the normal equations loses precision with the square of X's condition number, so nearly identical
    # templates, or two spikes of one unit at one time, need the fit from X itself. Column i's squared distance from
    # the span of the other columns is 1 / (G^-1)_ii; where it is at least 1e-8 of G_ii for every column, every
    # subsystem nnls solves is at least as well-conditioned, since fewer columns leave each one farther from the rest.
    try:
        L = np.linalg.cholesky(G)
    except Exception:
        return False
    inverse_L = np.linalg.solve(L, np.eye(G.shape[0]))
    for i in range(G.shape[0]):
        if 1.0 <= 1e-8 * G[i, i] * np.sum(inverse_L[:, i] ** 2):
            return False
    return True


@njit(cache=True)
def _solve_spd(G, b, idx):
    """Solve ``G[idx, idx] z = b[idx]`` for a well-conditioned ``G``."""
    # a Gram matrix is symmetric and positive definite once it is well-conditioned, so Cholesky applies
    L = np.linalg.cholesky(G[idx][:, idx])
    return np.linalg.solve(L.T, np.linalg.solve(L, b[idx]))


@njit(cache=True)
def _nnls_gram(G, b):
    """The ``x >= 0`` minimising ``|Xx - y|^2``, given a well-conditioned ``G = X'X`` and ``b = X'y``.

    This is the ``x`` that ``nnls(X, y)`` returns.
    """
    # Written out rather than calling scipy.optimize.nnls through numba.objmode: on systems this small, most of
    # nnls's time is per-call overhead, paid once per collision.
    #
    # Lawson-Hanson: coefficients start held at zero and are freed one at a time. Each pass solves the normal
    # equations for the free ones, and any that would go negative are returned to zero.
    k = b.size
    x = np.zeros(k)
    passive = np.zeros(k, dtype=np.bool_)  # True for the free coefficients
    scale = 0.0
    for i in range(k):
        scale = max(scale, abs(b[i]))
    tol = 1e-12 * scale
    idx = np.empty(k, dtype=np.int64)
    # the iteration caps stop rounding from cycling coefficients in and out forever
    for _ in range(3 * k):
        inner_steps = 0
        # w = b - Gx is minus the gradient of the error: free the held coefficient that lowers it fastest,
        # and stop when raising none of them would lower it
        best, best_w = -1, tol
        for j in range(k):
            if not passive[j]:
                w = b[j]
                for m in range(k):
                    w -= G[j, m] * x[m]
                if w > best_w:
                    best, best_w = j, w
        if best < 0:
            break
        passive[best] = True
        while inner_steps <= k:
            inner_steps += 1
            n = 0
            for j in range(k):
                if passive[j]:
                    idx[n] = j
                    n += 1
            z = _solve_spd(G, b, idx[:n])
            if np.all(z > 0.0):
                for i in range(n):
                    x[idx[i]] = z[i]
                break
            # z is the unconstrained fit on the free coefficients, but some are not positive:
            # step from x toward z until the first one reaches zero, hold it there and solve again
            alpha = 1.0
            for i in range(n):
                if z[i] <= 0.0:
                    alpha = min(alpha, x[idx[i]] / (x[idx[i]] - z[i]))
            for i in range(n):
                j = idx[i]
                x[j] += alpha * (z[i] - x[j])
                if x[j] <= 0.0:
                    x[j] = 0.0
                    passive[j] = False
    return x


@njit(cache=True, inline="always")
def _waveform(traces, t, c, gains, offsets):
    """One trace value, scaled by its channel's gain and offset unless ``gains`` is empty."""
    if gains.size > 0:
        # scaled in the gains' dtype before widening, so it rounds as the numpy path does
        return np.float64(traces[t, c] * gains[c] + offsets[c])
    return np.float64(traces[t, c])


@njit(cache=True)
def _ordinary_slope(traces, gains, offsets, sample, template, channels, cut_out_before):
    """Least-squares slope of the waveform on the template over ``channels``, or NaN for a constant template."""
    start = max(0, sample - cut_out_before)
    stop = min(traces.shape[0], sample - cut_out_before + template.shape[0])
    row = cut_out_before - sample
    count = channels.size * (stop - start)
    template_sum = 0.0
    waveform_sum = 0.0
    for t in range(start, stop):
        for c in channels:
            template_sum += template[row + t, c]
            waveform_sum += _waveform(traces, t, c, gains, offsets)
    template_mean = template_sum / count
    waveform_mean = waveform_sum / count
    variance = 0.0
    covariance = 0.0
    for t in range(start, stop):
        for c in channels:
            a = template[row + t, c] - template_mean
            variance += a * a
            covariance += a * (_waveform(traces, t, c, gains, offsets) - waveform_mean)
    # the node passes NaN to _ordinary_scaling_slope, so linregress decides a constant template, as in the numpy path
    if variance == 0.0:
        return np.nan
    return covariance / variance


@njit(cache=True)
def _template_terms(traces, gains, offsets, sample, template, channels, cut_out_before):
    """Sum of a spike's template over ``channels``, its sum of squares, and its dot product with the waveform."""
    start = max(0, sample - cut_out_before)
    stop = min(traces.shape[0], sample - cut_out_before + template.shape[0])
    row = cut_out_before - sample
    total = 0.0
    squares = 0.0
    product = 0.0
    for t in range(start, stop):
        for c in channels:
            a = np.float64(template[row + t, c])
            total += a
            squares += a * a
            product += a * _waveform(traces, t, c, gains, offsets)
    return total, squares, product


@njit(cache=True)
def _collision_scaling(
    traces,
    gains,
    offsets,
    samples,
    units,
    templates,
    sparsity_mask,
    sparse_starts,
    sparse_lists,
    spike_terms,
    cut_out_before,
):
    """Scaling of the first spike in a non-negative fit of all the spikes' templates, plus an intercept.

    ``spike_terms`` holds each spike's ``_template_terms``. Returns NaN where the fit should be done from ``X``: for
    traces that are not finite, so the numpy path raises on them, and for an ill-conditioned fit.
    """
    num_samples, num_channels = traces.shape
    template_length = templates.shape[1]
    k = samples.size

    # the rows of X, as fit_collision cuts them: every sample of the window on any of the spikes' channels
    union = np.zeros(num_channels, dtype=np.bool_)
    for i in range(k):
        union |= sparsity_mask[units[i]]
    union_channels = np.flatnonzero(union)
    window_start = max(0, samples.min() - cut_out_before)
    window_end = min(num_samples, samples.max() - cut_out_before + template_length)
    y_sum = 0.0
    for t in range(window_start, window_end):
        for c in union_channels:
            y_sum += _waveform(traces, t, c, gains, offsets)
    if not np.isfinite(y_sum):
        return np.nan

    # the normal equations of fit_collision's X, built without X
    col_sums = spike_terms[:, 0].copy()
    Xy = spike_terms[:, 2].copy()
    G = np.zeros((k, k))
    shared = np.empty(num_channels, dtype=np.int64)
    for i in range(k):
        G[i, i] = spike_terms[i, 1]
        ti = templates[units[i]]
        row_i = cut_out_before - samples[i]
        channels_i = sparse_lists[sparse_starts[units[i]] : sparse_starts[units[i] + 1]]
        for j in range(i + 1, k):
            tj = templates[units[j]]
            row_j = cut_out_before - samples[j]
            start = max(window_start, samples[i] - cut_out_before, samples[j] - cut_out_before)
            stop = min(
                window_end, samples[i] - cut_out_before + template_length, samples[j] - cut_out_before + template_length
            )
            # x_i'x_j: templates are zero outside their sparsity, so only shared samples and channels contribute
            num_shared = 0
            for c in channels_i:
                if sparsity_mask[units[j], c]:
                    shared[num_shared] = c
                    num_shared += 1
            s = 0.0
            for t in range(start, stop):
                for c in shared[:num_shared]:
                    s += np.float64(ti[row_i + t, c]) * tj[row_j + t, c]
            G[i, j] = s

    # fit_collision fits the intercept by subtracting each column's mean and the waveform's mean; in terms of
    # the column sums s and the n rows, that turns X'X into X'X - s s'/n and X'y into X'y - s sum(y)/n
    num_rows = (window_end - window_start) * union_channels.size
    for i in range(k):
        Xy[i] -= col_sums[i] * y_sum / num_rows
        for j in range(i, k):
            G[i, j] -= col_sums[i] * col_sums[j] / num_rows
            G[j, i] = G[i, j]
    if not _is_well_conditioned(G):
        return np.nan
    return _nnls_gram(G, Xy)[0]


@njit(cache=True)
def compute_scalings(
    traces,
    gains,
    offsets,
    sample_indices,
    unit_indices,
    local_spike_indices,
    overlap_matrix,
    delta_collision_samples,
    templates,
    sparsity_mask,
    sparse_starts,
    sparse_lists,
    template_is_zero,
    cut_out_before,
):
    """Scaling of each spike in ``local_spike_indices``, whether it collides, and whether the numpy path must fit it.

    ``templates`` are ``(units, samples, channels)``, cut to the fitted window, and zero outside ``sparsity_mask``,
    whose channels are listed per unit in ``sparse_lists[sparse_starts[u] : sparse_starts[u + 1]]``.
    ``sample_indices`` and ``unit_indices`` cover the chunk and its margin, and a negative
    ``delta_collision_samples`` fits every spike alone. ``gains`` of size 0 leave the traces unscaled.
    """
    num_peaks = sample_indices.size
    # a spike's column sum, x_i'x_i and x_i'y depend only on the spike, so they are computed once and reused in
    # every collision it is in
    spike_terms = np.zeros((num_peaks, 3))
    if delta_collision_samples >= 0:
        for p in range(num_peaks):
            u = unit_indices[p]
            channels = sparse_lists[sparse_starts[u] : sparse_starts[u + 1]]
            total, squares, product = _template_terms(
                traces, gains, offsets, sample_indices[p], templates[u], channels, cut_out_before
            )
            spike_terms[p, 0], spike_terms[p, 1], spike_terms[p, 2] = total, squares, product

    num_spikes = local_spike_indices.size
    scalings = np.zeros(num_spikes)
    is_collision = np.zeros(num_spikes, dtype=np.bool_)
    needs_numpy_fit = np.zeros(num_spikes, dtype=np.bool_)
    for spike in range(num_spikes):
        center = local_spike_indices[spike]
        sample = sample_indices[center]
        unit = unit_indices[center]
        # as in find_collisions: spikes within delta_collision_samples, of units sharing a channel with this one
        lo = hi = 0
        if delta_collision_samples >= 0:
            lo = np.searchsorted(sample_indices, sample - delta_collision_samples)
            hi = np.searchsorted(sample_indices, sample + delta_collision_samples)
        members = np.empty(hi - lo + 1, dtype=np.int64)
        members[0] = center
        k = 1
        for other in range(lo, hi):
            if other != center and overlap_matrix[unit, unit_indices[other]]:
                members[k] = other
                k += 1
        if k > 1:
            members = members[:k]
            is_collision[spike] = True
            scaling = _collision_scaling(
                traces,
                gains,
                offsets,
                sample_indices[members],
                unit_indices[members],
                templates,
                sparsity_mask,
                sparse_starts,
                sparse_lists,
                spike_terms[members],
                cut_out_before,
            )
            if np.isnan(scaling):
                needs_numpy_fit[spike] = True
            else:
                scalings[spike] = scaling
        elif not template_is_zero[unit]:  # a zero template keeps a scaling of 0, as in the numpy path
            channels = sparse_lists[sparse_starts[unit] : sparse_starts[unit + 1]]
            slope = _ordinary_slope(traces, gains, offsets, sample, templates[unit], channels, cut_out_before)
            if np.isnan(slope):
                needs_numpy_fit[spike] = True
            else:
                scalings[spike] = slope
    return scalings, is_collision, needs_numpy_fit
