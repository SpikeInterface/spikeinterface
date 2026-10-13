"""numba kernels for ``_row_median``, kept at module level so that ``cache=True`` can find them again.

Both kernels return exactly what ``np.median(x, axis=1)`` returns. For a row of ``C`` values, sorted, the median
is the mean of the values at ranks ``r_lo = (C - 1) // 2`` and ``r_hi = C // 2``, which are the same rank when
``C`` is odd. Neither kernel sorts the whole row in the common case; each finds those two values another way.

``int_row_median`` counts each row into a histogram. Filtered integer traces take few distinct values, and the
median moves little from one sample to the next, so a window of bins centred on the previous row's median
almost always holds both middle values. A cumulative sum over the bins then gives them directly.

``float_row_median`` narrows an interval ``[a, b]`` known to hold both middle values. Each pass counts how many
values fall below a probe ``p`` and moves ``a`` or ``b`` to ``p``. Once the interval holds few values, those are
gathered and sorted. The counting loop has no data-dependent branch, so it vectorises.

Rows these shortcuts cannot handle are sorted in full, so every path returns the exact median.
"""

import numpy as np
from numba import njit

# The float interval is sorted once it holds this many values; smaller needs more counting passes than it saves.
_FINAL_WIDTH = 24
# Counting passes before giving up and sorting the row; real traces need about three.
_MAX_PASSES = 48


@njit(nogil=True, cache=True)
def int_row_median(x, window):
    """Median of each row of integer ``x`` (at most 16 bits), as float64 like ``np.median``.

    ``window`` is the number of histogram bins. Bin ``i`` counts the value ``base + i``, and ``base`` is chosen
    per row so that the previous row's median sits in the middle bin.
    """
    T, C = x.shape
    out = np.empty(T, dtype=np.float64)
    counts = np.zeros(window, dtype=np.int32)
    r_lo = (C - 1) // 2  # the two middle ranks, equal for odd C
    r_hi = C // 2
    center = np.int64(np.sort(x[0])[r_lo])
    for t in range(T):
        row = x[t]
        base = center - window // 2
        # Count every value into its bin. Values below the window are only counted, since the rank of anything
        # inside the window depends on how many lie below it; values above it can be ignored.
        below = 0
        vmin = np.int64(window)  # lowest occupied bin, where the cumulative scan starts
        for c in range(C):
            i = np.int64(row[c]) - base
            if i < 0:
                below += 1
            elif i < window:
                counts[i] += 1
                vmin = min(vmin, i)
        # Walk the bins in value order, keeping a running count of values seen. The value at rank r is the
        # first bin where that count exceeds r. If more than r_lo values lie below the window, the middle
        # values are below it too and the scan is skipped; if the scan ends without reaching r_hi, they are above.
        lo = -1
        hi = -1
        if below <= r_lo:
            cum = below
            for i in range(vmin, window):
                cum += counts[i]
                if lo < 0 and cum > r_lo:
                    lo = i
                if cum > r_hi:
                    hi = i
                    break
        for c in range(C):  # clear only the bins this row touched, which is cheaper than zeroing the window
            i = np.int64(row[c]) - base
            if 0 <= i < window:
                counts[i] = 0
        if hi >= 0:
            out[t] = 0.5 * (np.float64(lo + base) + np.float64(hi + base))
            center = lo + base
        else:
            # A middle value fell outside the window, as after a large jump in the median: sort, and recentre
            # the window on this row's median for the next row.
            s = np.sort(row)
            out[t] = 0.5 * (np.float64(s[r_lo]) + np.float64(s[r_hi]))
            center = np.int64(s[r_lo])
    return out


@njit(inline="always")
def _counts(row, v):
    """Numbers of values in ``row`` below ``v`` and at most ``v``.

    The comparisons are summed rather than branched on, so the loop compiles to vector compares.
    """
    lt = 0
    le = 0
    for c in range(row.size):
        lt += row[c] < v
        le += row[c] <= v
    return lt, le


@njit(inline="always")
def _sorted_median(row, scratch, r_lo, r_hi):
    """Median by sorting a copy of ``row``; NaN if the row holds one, as ``np.median`` returns."""
    scratch[:] = row
    scratch.sort()  # NaN sorts last
    if np.isnan(scratch[-1]):
        return scratch[-1]
    return np.float32(0.5) * (scratch[r_lo] + scratch[r_hi])


@njit(inline="always")
def _median_of_row(row, buf, scratch):
    """Median of one float32 row, by narrowing an interval around the middle ranks.

    ``buf`` (at least ``C + 1`` values) and ``scratch`` (``C`` values) are work space reused across rows.
    """
    C = row.size
    r_lo = (C - 1) // 2  # the two middle ranks, equal for odd C
    r_hi = C // 2
    total = np.float32(0.0)
    vmin = row[0]
    vmax = row[0]
    for c in range(C):
        v = row[c]
        total += v
        vmin = min(vmin, v)
        vmax = max(vmax, v)
    if not np.isfinite(total):  # NaN, inf, or a sum that overflowed
        return _sorted_median(row, scratch, r_lo, r_hi)
    # Invariant: ca = count(< a) <= r_lo and cb = count(<= b) >= r_hi + 1, so both middle values lie in
    # [a, b], and cb - ca of the row's values lie there.
    a = vmin
    ca = 0
    b = vmax
    cb = C
    p = total / C  # the mean is usually close to the median of filtered traces
    target = 0.5 * (r_lo + r_hi + 1)  # the count below the median, where the probe aims
    for _ in range(_MAX_PASSES):
        if cb - ca <= _FINAL_WIDTH:
            break
        lt, le = _counts(row, p)
        if lt <= r_lo and le >= r_hi + 1:
            return p  # p is a value holding both middle ranks
        if lt > r_lo and le < r_hi + 1:
            # Only possible for even C, with exactly r_hi values below p and none equal to it: the middle values
            # are then the largest value below p and the smallest above it.
            below = vmin
            above = vmax
            for c in range(C):
                v = row[c]
                below = max(below, v if v < p else vmin)
                above = min(above, v if v > p else vmax)
            return np.float32(0.5) * (below + above)
        # p is below both middle values or above both; move that end of the interval to p.
        if lt <= r_lo:
            a = p
            ca = lt
        else:
            b = p
            cb = le
        # Next probe: interpolate linearly between (a, ca) and (b, cb) to where the count reaches target, as
        # regula falsi does on the row's empirical CDF. Fall back to bisection if rounding puts it outside (a, b).
        q = a + (target - ca) / max(cb - ca, 1) * (b - a)
        if not (a < q < b):
            q = 0.5 * (a + b)
        p = np.float32(q)
    if cb - ca > _FINAL_WIDTH:
        return _sorted_median(row, scratch, r_lo, r_hi)
    # Gather the values in [a, b] without branching: every value is written, and the write position advances
    # only past values in the interval. buf has one spare slot for the last write.
    n = 0
    for c in range(C):
        v = row[c]
        buf[n] = v
        n += (v >= a) & (v <= b)
    middle = buf[:n]
    middle.sort()
    # ca values lie below a, so rank r in the row is rank r - ca among the gathered values.
    return np.float32(0.5) * (middle[r_lo - ca] + middle[r_hi - ca])


# Reassociation only reorders the sum that seeds the search, so the result stays exact.
@njit(nogil=True, cache=True, fastmath={"reassoc"})
def float_row_median(x):
    """Median of each row of float32 ``x``, as float32 like ``np.median``."""
    T, C = x.shape
    out = np.empty(T, dtype=np.float32)
    buf = np.empty(C + 1, dtype=np.float32)
    scratch = np.empty(C, dtype=np.float32)
    for t in range(T):
        out[t] = _median_of_row(x[t], buf, scratch)
    return out
