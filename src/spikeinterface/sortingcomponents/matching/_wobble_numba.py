"""Numba kernels for wobble's joint fit of overlapping spikes (see wobble._JointFit).

The kernels' arguments include three tuples:
- state: the residual (channels by samples, float32); each spike's start, jittered template index and amplitude;
  the spikes in each bin of num_samples samples, as linked lists (head, tail and next); whether each spike changed
  since the objectives were last updated and, if so, the start, jittered template index and amplitude they have for
  it; the number of spikes and of changed spikes; and the changed spikes in order;
- templates: each jittered template's waveform on its template's visible channels (padded to the most visible
  channels of any template), those channels and how many there are, each channel's position among a template's
  visible channels (-1 where it is not visible), the squared norms, the template overlap and each template's unit;
- params: jitter factor, 1 / amplitude variance, scale bounds, refractory period, re-placements per spike and the most
  spikes a spike is fitted together with.
"""

import numpy as np
from numba import njit

# cache=True keeps the compiled code on disk, so it compiles on first use rather than in every session.


@njit(cache=True, fastmath={"reassoc", "contract"})
def add_spikes(spike_train, state, templates, params):
    """Place and fit new spikes (one batch, too far apart to overlap), taking out and placing again the spikes each
    overlaps while any of them moves; their rows of spike_train are updated."""
    starts, jittered = state[1], state[2]
    max_replacements = params[5]
    for row in range(spike_train.shape[0]):
        slot, near = _place(spike_train[row, 0], spike_train[row, 1], -1, state, templates, params)
        involved = [np.int64(i) for i in near]
        involved.append(slot)
        queue = [np.int64(i) for i in near]
        budget = max_replacements * len(involved)
        while len(queue) > 0 and budget > 0:
            i = queue.pop(0)
            budget -= 1
            start, jittered_index = starts[i], jittered[i]
            _change(i, 0.0, state, templates, params)
            _place(start, jittered_index, i, state, templates, params)
            if starts[i] != start or jittered[i] != jittered_index:
                for j in involved:
                    if j != i and j not in queue:
                        queue.append(j)
        spike_train[row, 0] = starts[slot]
        spike_train[row, 1] = jittered[slot]


@njit(cache=True, fastmath={"reassoc", "contract"})
def _place(start, jittered_index, slot, state, templates, params):
    """Refine a spike's start and jitter, then fit it with the spikes it overlaps (at most the params[6] most coupled to
    it); returns its index and those."""
    num_samples = templates[0].shape[2]
    near = _overlapping(start, jittered_index, num_samples + 1, slot, state, templates, params)
    if len(near) > params[6]:
        near = _most_coupled(start, jittered_index, near, state, templates, params)
    gram, rhs, amplitudes = _near_system(near, state, templates, params)
    start, jittered_index = _refine(start, jittered_index, near, gram, rhs, amplitudes, state, templates, params)
    slot = _fit(start, jittered_index, near, gram, rhs, amplitudes, slot, state, templates, params)
    return slot, near


@njit(cache=True, fastmath={"reassoc", "contract"})
def _overlapping(start, jittered_index, reach, exclude, state, templates, params):
    """Spikes other than exclude starting less than reach samples from start, on channels the template shares."""
    starts, jittered, bin_head, bin_next = state[1], state[2], state[4], state[6]
    unit_overlap, jitter_factor = templates[5], params[0]
    num_samples = templates[0].shape[2]
    found = np.empty(0, dtype=np.int64)
    for counting in (True, False):
        count = 0
        center = start // num_samples
        for b in range(max(center - 1, 0), min(center + 2, bin_head.shape[0])):
            i = bin_head[b]
            while i >= 0:
                if (
                    i != exclude
                    and abs(starts[i] - start) < reach
                    and unit_overlap[jittered_index, jittered[i] // jitter_factor]
                ):
                    if not counting:
                        found[count] = i
                    count += 1
                i = bin_next[i]
        if counting:
            found = np.empty(count, dtype=np.int64)
    return found


@njit(cache=True, fastmath={"reassoc", "contract"})
def _most_coupled(start, jittered_index, near, state, templates, params):
    """The spikes of near whose amplitudes are most coupled to the spike's (their inner product over the smaller squared
    norm), as many as params allows, with any of the same unit within the refractory period first; in their order."""
    starts, jittered = state[1], state[2]
    norms, unit_of = templates[4], templates[6]
    jitter_factor, inv_lambda, refractory_period = params[0], params[1], params[4]
    unit = unit_of[jittered_index // jitter_factor]
    coupling = np.empty(len(near))
    for p in range(len(near)):
        i = near[p]
        if unit_of[jittered[i] // jitter_factor] == unit and abs(starts[i] - start) <= refractory_period:
            coupling[p] = np.inf
        else:
            product = _template_product(start, jittered_index, starts[i], jittered[i], templates, params)
            coupling[p] = abs(product) / (min(norms[jittered_index], norms[jittered[i]]) + inv_lambda)
    kept = np.sort(np.argsort(-coupling, kind="mergesort")[: params[6]])
    return near[kept]


@njit(cache=True, fastmath={"reassoc", "contract"})
def _residual_product(start, jittered_index, state, templates, params):
    """Inner product of a jittered template starting at start with the residual, which is zero outside the chunk."""
    residual = state[0]
    waveforms, channels, num_visible = templates[0], templates[1], templates[2]
    template_index = jittered_index // params[0]
    num_samples = waveforms.shape[2]
    lo, hi = max(0, -start), min(num_samples, residual.shape[1] - start)
    total = 0.0
    for p in range(num_visible[template_index]):
        # sliced first, so that indices are known to be non-negative
        waveform = waveforms[jittered_index, p, lo:hi]
        trace = residual[channels[template_index, p], start + lo : start + hi]
        for t in range(hi - lo):
            total += np.float64(waveform[t]) * trace[t]
    return total


@njit(cache=True, fastmath={"reassoc", "contract"})
def _template_product(start, jittered_index, other_start, other_jittered_index, templates, params):
    """Inner product of two jittered templates at their starts."""
    waveforms, channels, num_visible, channel_position = templates[0], templates[1], templates[2], templates[3]
    num_samples = waveforms.shape[2]
    lag = other_start - start
    if lag <= -num_samples or lag >= num_samples:
        return 0.0
    template_index, other_template_index = jittered_index // params[0], other_jittered_index // params[0]
    lo, hi = max(0, lag), min(num_samples, num_samples + lag)
    total = 0.0
    for p in range(num_visible[template_index]):
        q = channel_position[other_template_index, channels[template_index, p]]
        if q < 0:
            continue
        waveform = waveforms[jittered_index, p, lo:hi]
        other_waveform = waveforms[other_jittered_index, q, lo - lag : hi - lag]
        for t in range(hi - lo):
            total += np.float64(waveform[t]) * other_waveform[t]
    return total


@njit(cache=True, fastmath={"reassoc", "contract"})
def _near_system(near, state, templates, params):
    """Gram matrix (with the prior) of the spikes near, their least-squares right-hand side with them added back to
    the residual, and their amplitudes."""
    starts, jittered, amplitudes = state[1], state[2], state[3]
    inv_lambda = params[1]
    k = len(near)
    gram = np.zeros((k, k))
    rhs = np.empty(k)
    near_amplitudes = np.empty(k)
    for p in range(k):
        i = near[p]
        near_amplitudes[p] = amplitudes[i]
        rhs[p] = _residual_product(starts[i], jittered[i], state, templates, params)
        for q in range(p, k):
            j = near[q]
            gram[p, q] = _template_product(starts[i], jittered[i], starts[j], jittered[j], templates, params)
            gram[q, p] = gram[p, q]
    for p in range(k):
        rhs[p] += inv_lambda
        for q in range(k):
            rhs[p] += gram[p, q] * near_amplitudes[q]
    for p in range(k):
        gram[p, p] += inv_lambda
    return gram, rhs, near_amplitudes


@njit(cache=True, fastmath={"reassoc", "contract"})
def _refine(start, jittered_index, near, gram, rhs, near_amplitudes, state, templates, params):
    """The start (within a sample) and jitter with the largest fitted gain, the spikes near refitted with it, that is
    not within the refractory period of a spike of the same unit."""
    residual, starts, jittered = state[0], state[1], state[2]
    norms, unit_of = templates[4], templates[6]
    jitter_factor, inv_lambda, refractory_period = params[0], params[1], params[4]
    num_samples = templates[0].shape[2]
    template_index = jittered_index // jitter_factor
    first_jittered = template_index * jitter_factor
    first = start - 1
    corr = np.empty((3, jitter_factor))
    denominator = np.empty((3, jitter_factor))
    allowed = np.empty(3, dtype=np.bool_)
    for o in range(3):
        allowed[o] = first + o >= 0 and first + o + num_samples <= residual.shape[1]
        for j in range(jitter_factor):
            corr[o, j] = _residual_product(first + o, first_jittered + j, state, templates, params)
            denominator[o, j] = norms[first_jittered + j] + inv_lambda
    k = len(near)
    if k > 0:
        gram_inverse = np.linalg.inv(gram)
        change = near_amplitudes.copy()
        for p in range(k):
            for q in range(k):
                change[p] -= gram_inverse[p, q] * rhs[q]
        # inner products of every candidate with each spike near
        cross = np.zeros((3, jitter_factor, k))
        for q in range(k):
            i = near[q]
            same_unit = unit_of[jittered[i] // jitter_factor] == unit_of[template_index]
            for o in range(3):
                lag = starts[i] - (first + o)
                if same_unit and abs(lag) < refractory_period:
                    allowed[o] = False
                if abs(lag) < num_samples:
                    for j in range(jitter_factor):
                        cross[o, j, q] = _template_product(
                            first + o, first_jittered + j, starts[i], jittered[i], templates, params
                        )
        for o in range(3):
            for j in range(jitter_factor):
                for p in range(k):
                    corr[o, j] += cross[o, j, p] * change[p]
                    quadratic = 0.0
                    for q in range(k):
                        quadratic += gram_inverse[p, q] * cross[o, j, q]
                    denominator[o, j] -= cross[o, j, p] * quadratic
    best, best_offset, best_jitter = -np.inf, -1, -1
    for o in range(3):
        if not allowed[o]:
            continue
        for j in range(jitter_factor):
            c = corr[o, j] + inv_lambda
            if c > 0 and denominator[o, j] > 1e-6 * norms[first_jittered + j]:
                score = c * c / denominator[o, j]
                if score > best:
                    best, best_offset, best_jitter = score, o, j
    if best_offset < 0:
        return start, jittered_index
    return first + best_offset, first_jittered + best_jitter


@njit(cache=True, fastmath={"reassoc", "contract"})
def _fit(start, jittered_index, near, gram, rhs, near_amplitudes, slot, state, templates, params):
    """Fit a spike (new, or taken out at slot) and the spikes near together, update the residual, and record it;
    returns its index."""
    starts, jittered, bin_head, bin_tail, bin_next, counts = state[1], state[2], state[4], state[5], state[6], state[11]
    norms = templates[4]
    inv_lambda, scale_min, scale_max = params[1], params[2], params[3]
    num_samples = templates[0].shape[2]
    k = len(near)
    new_rhs = _residual_product(start, jittered_index, state, templates, params) + inv_lambda
    if k > 0:
        full_gram = np.empty((k + 1, k + 1))
        full_gram[:k, :k] = gram
        full_rhs = np.empty(k + 1)
        full_rhs[:k] = rhs
        for q in range(k):
            i = near[q]
            cross = _template_product(start, jittered_index, starts[i], jittered[i], templates, params)
            full_gram[q, k] = cross
            full_gram[k, q] = cross
            # the new spike's inner product with the residual, the spikes near added back
            new_rhs += cross * near_amplitudes[q]
        full_gram[k, k] = norms[jittered_index] + inv_lambda
        full_rhs[k] = new_rhs
        fitted = _bounded_solve(full_gram, full_rhs, scale_min, scale_max)
    else:
        fitted = np.array([min(max(new_rhs / (norms[jittered_index] + inv_lambda), scale_min), scale_max)])
    for q in range(k):
        _change(near[q], fitted[q], state, templates, params)
    if slot < 0:
        slot = counts[0]
        counts[0] += 1
        starts[slot], jittered[slot], state[3][slot] = start, jittered_index, 0.0
        _bin_append(slot, start // num_samples, bin_head, bin_tail, bin_next)
        _record(slot, state)
    else:
        # taken out, so at amplitude 0
        if start // num_samples != starts[slot] // num_samples:
            _bin_remove(slot, starts[slot] // num_samples, bin_head, bin_tail, bin_next)
            _bin_append(slot, start // num_samples, bin_head, bin_tail, bin_next)
        starts[slot], jittered[slot] = start, jittered_index
    _change(slot, fitted[k], state, templates, params)
    return slot


@njit(cache=True, fastmath={"reassoc", "contract"})
def _change(i, amplitude, state, templates, params):
    """Set a spike's amplitude and update the residual (only)."""
    residual, starts, jittered, amplitudes = state[0], state[1], state[2], state[3]
    waveforms, channels, num_visible = templates[0], templates[1], templates[2]
    change = amplitude - amplitudes[i]
    if change == 0:
        return
    _record(i, state)
    template_index = jittered[i] // params[0]
    start, num_samples = starts[i], waveforms.shape[2]
    scale = np.float32(change)
    for p in range(num_visible[template_index]):
        waveform = waveforms[jittered[i], p]
        trace = residual[channels[template_index, p], start : start + num_samples]
        for t in range(num_samples):
            trace[t] -= scale * waveform[t]
    amplitudes[i] = amplitude


@njit(cache=True, fastmath={"reassoc", "contract"})
def _record(i, state):
    """Keep a spike's start, jittered index and amplitude as the objectives have them, if not kept already."""
    stale, counts, changed = state[7], state[11], state[12]
    if not stale[i]:
        stale[i] = True
        state[8][i], state[9][i], state[10][i] = state[1][i], state[2][i], state[3][i]
        changed[counts[1]] = i
        counts[1] += 1


@njit(cache=True, fastmath={"reassoc", "contract"})
def _bin_append(i, b, bin_head, bin_tail, bin_next):
    bin_next[i] = -1
    if bin_head[b] < 0:
        bin_head[b] = i
    else:
        bin_next[bin_tail[b]] = i
    bin_tail[b] = i


@njit(cache=True, fastmath={"reassoc", "contract"})
def _bin_remove(i, b, bin_head, bin_tail, bin_next):
    previous, j = -1, bin_head[b]
    while j != i:
        previous, j = j, bin_next[j]
    if previous < 0:
        bin_head[b] = bin_next[i]
    else:
        bin_next[previous] = bin_next[i]
    if bin_tail[b] == i:
        bin_tail[b] = previous


@njit(cache=True, fastmath={"reassoc", "contract"})
def _bounded_solve(gram, rhs, lower, upper):
    """Solve gram @ x = rhs for lower <= x <= upper: entries out of bounds are fixed at them and the rest solved
    again."""
    x = np.linalg.solve(gram, rhs)
    m = len(x)
    fixed = np.zeros(m, dtype=np.bool_)
    while True:
        out = False
        for i in range(m):
            if not fixed[i] and (x[i] < lower or x[i] > upper):
                x[i] = min(max(x[i], lower), upper)
                fixed[i] = True
                out = True
        if not out:
            return x
        free = np.flatnonzero(~fixed)
        if len(free) == 0:
            return x
        sub_gram = np.empty((len(free), len(free)))
        sub_rhs = np.empty(len(free))
        for a in range(len(free)):
            sub_rhs[a] = rhs[free[a]]
            for i in range(m):
                if fixed[i]:
                    sub_rhs[a] -= gram[free[a], i] * x[i]
            for b in range(len(free)):
                sub_gram[a, b] = gram[free[a], free[b]]
        solved = np.linalg.solve(sub_gram, sub_rhs)
        for a in range(len(free)):
            x[free[a]] = solved[a]


# The objectives are updated with the same float32 operations as subtract_spike_train uses, so without fastmath.


@njit(cache=True)
def sync_objectives(objective, objective_normalized, state, convolutions):
    """Update the objectives for the spikes changed since the last update."""
    starts, jittered, amplitudes = state[1], state[2], state[3]
    stale, synced_starts, synced_jittered, synced_amplitudes = state[7], state[8], state[9], state[10]
    counts, changed = state[11], state[12]
    for c in range(counts[1]):
        i = changed[c]
        stale[i] = False
        if synced_starts[i] == starts[i] and synced_jittered[i] == jittered[i]:
            _subtract(
                objective,
                objective_normalized,
                starts[i],
                jittered[i],
                amplitudes[i] - synced_amplitudes[i],
                convolutions,
            )
        else:
            _subtract(
                objective,
                objective_normalized,
                synced_starts[i],
                synced_jittered[i],
                -synced_amplitudes[i],
                convolutions,
            )
            _subtract(objective, objective_normalized, starts[i], jittered[i], amplitudes[i], convolutions)
    counts[1] = 0


@njit(cache=True)
def _subtract(objective, objective_normalized, start, jittered_index, change, convolutions):
    """Subtract a jittered template's pairwise convolutions at change times its amplitude from the objectives."""
    if change == 0:
        return
    pairwise_convolution, overlapping, num_overlapping = convolutions
    scale, scale_normalized = np.float32(change), np.float32(2 * change)
    for k in range(num_overlapping[jittered_index]):
        convolution = pairwise_convolution[jittered_index, k]
        row = objective[overlapping[jittered_index, k], start : start + len(convolution)]
        row_normalized = objective_normalized[overlapping[jittered_index, k], start : start + len(convolution)]
        for t in range(len(convolution)):
            row[t] -= scale * convolution[t]
            row_normalized[t] -= scale_normalized * convolution[t]


@njit(cache=True)
def credit_projections(spike_train, projection, objective_normalized, convolutions, norms, inv_lambda):
    """Credit the templates near each new spike with their projection onto it, where it is the largest yet."""
    pairwise_convolution, overlapping, num_overlapping = convolutions
    for row in range(spike_train.shape[0]):
        start, jittered_index = spike_train[row, 0], spike_train[row, 1]
        denominator = np.float32(norms[jittered_index] + inv_lambda)
        for k in range(num_overlapping[jittered_index]):
            convolution = pairwise_convolution[jittered_index, k]
            largest = projection[overlapping[jittered_index, k], start : start + len(convolution)]
            row_normalized = objective_normalized[overlapping[jittered_index, k], start : start + len(convolution)]
            for t in range(len(convolution)):
                increase = convolution[t] * convolution[t] / denominator - largest[t]
                if increase > 0:
                    largest[t] += increase
                    row_normalized[t] += increase
