"""Amplitude scalings of a chunk's spikes, with collisions solved on the k x k normal equations of k spikes.

``fit_collision`` builds a design matrix with one row per sample and channel of the collision window, mostly
zeros, then centres it and calls ``nnls``. Least squares depends on that matrix only through its Gram matrix
and its product with the waveform. When every template is zero outside its unit's channels, each entry of those
is a product of one or two templates over the channels and samples they share, so a spike's own terms are
computed once and shared by every collision it is in.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _solve_spd(G, b, idx, n):
    """Solve ``G[idx, idx] z = b[idx]`` by Cholesky; returns False if the subsystem is singular."""
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = G[idx[i], idx[j]]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-12 * abs(G[idx[i], idx[i]]) or s <= 0.0:
                    return False, L[0]
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    z = np.zeros(n)
    for i in range(n):
        s = b[idx[i]]
        for k in range(i):
            s -= L[i, k] * z[k]
        z[i] = s / L[i, i]
    for i in range(n - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, n):
            s -= L[k, i] * z[k]
        z[i] = s / L[i, i]
    return True, z


@njit(cache=True)
def _nnls_gram(G, b):
    """Lawson-Hanson NNLS on the normal equations: minimise ``x'Gx/2 - b'x`` subject to ``x >= 0``."""
    k = b.size
    x = np.zeros(k)
    passive = np.zeros(k, dtype=np.bool_)
    scale = 0.0
    for i in range(k):
        scale = max(scale, abs(b[i]))
    tol = 1e-12 * scale
    idx = np.empty(k, dtype=np.int64)
    for _ in range(3 * k):
        inner_steps = 0
        # gradient of the objective, restricted to the active (zero) set
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
            ok, z = _solve_spd(G, b, idx, n)
            if not ok:
                # a dependent column: drop the one just added and stop, as no direction improves the fit
                passive[best] = False
                return x
            if np.all(z > 0.0):
                for i in range(n):
                    x[idx[i]] = z[i]
                break
            # step from x toward z until the first passive coefficient reaches zero
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
    """One trace value as the numpy path fits it: scaled in the gains' dtype when ``gains`` is not empty."""
    if gains.size > 0:
        return np.float64(traces[t, c] * gains[c] + offsets[c])
    return np.float64(traces[t, c])


@njit(cache=True)
def _ordinary_slope(traces, gains, offsets, sample, template, channels, cut_out_before):
    """``linregress`` slope of the waveform on the template over ``channels``, or NaN for a constant template."""
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
    if variance == 0.0:
        return np.nan
    return covariance / variance


@njit(cache=True)
def _template_terms(traces, gains, offsets, sample, template, channels, cut_out_before):
    """Sum, sum of squares, and product with the waveform of one spike's template over ``channels``."""
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
    """Scaling of the first spike in a non-negative fit of all the spikes' templates plus an intercept."""
    num_samples, num_channels = traces.shape
    template_length = templates.shape[1]
    k = samples.size

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

    # centring the columns and the waveform fits the intercept
    num_rows = (window_end - window_start) * union_channels.size
    for i in range(k):
        Xy[i] -= col_sums[i] * y_sum / num_rows
        for j in range(i, k):
            G[i, j] -= col_sums[i] * col_sums[j] / num_rows
            G[j, i] = G[i, j]
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
    """Scaling of each spike in ``local_spike_indices``, whether it collides, and whether it needs ``linregress``.

    ``templates`` are ``(units, samples, channels)``, cut to the fitted window, and zero outside ``sparsity_mask``,
    whose channels are listed per unit in ``sparse_lists[sparse_starts[u] : sparse_starts[u + 1]]``.
    ``sample_indices`` and ``unit_indices`` cover the chunk and its margin, and a negative
    ``delta_collision_samples`` fits every spike alone. ``gains`` of size 0 leave the traces unscaled.
    """
    num_peaks = sample_indices.size
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
    needs_linregress = np.zeros(num_spikes, dtype=np.bool_)
    for spike in range(num_spikes):
        center = local_spike_indices[spike]
        sample = sample_indices[center]
        unit = unit_indices[center]
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
            scalings[spike] = _collision_scaling(
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
        elif not template_is_zero[unit]:
            channels = sparse_lists[sparse_starts[unit] : sparse_starts[unit + 1]]
            slope = _ordinary_slope(traces, gains, offsets, sample, templates[unit], channels, cut_out_before)
            if np.isnan(slope):
                needs_linregress[spike] = True
            else:
                scalings[spike] = slope
    return scalings, is_collision, needs_linregress
