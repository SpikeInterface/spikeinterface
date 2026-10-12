"""numba kernels for ``_row_median``, kept at module level so that ``cache=True`` can find them again."""

import numpy as np
from numba import njit

# The interval is sorted once it holds this many values; smaller needs more counting passes than it saves.
_FINAL_WIDTH = 24
_MAX_PASSES = 48


@njit(nogil=True, cache=True)
def int_row_median(x, window):
    T, C = x.shape
    out = np.empty(T, dtype=np.float64)
    counts = np.zeros(window, dtype=np.int32)
    r_lo = (C - 1) // 2  # the two middle ranks, equal for odd C
    r_hi = C // 2
    center = np.int64(np.sort(x[0])[r_lo])
    for t in range(T):
        row = x[t]
        base = center - window // 2
        below = 0
        vmin = np.int64(window)
        for c in range(C):
            i = np.int64(row[c]) - base
            if i < 0:
                below += 1
            elif i < window:
                counts[i] += 1
                vmin = min(vmin, i)
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
        for c in range(C):  # clear only the bins this row touched
            i = np.int64(row[c]) - base
            if 0 <= i < window:
                counts[i] = 0
        if hi >= 0:
            out[t] = 0.5 * (np.float64(lo + base) + np.float64(hi + base))
            center = lo + base
        else:
            s = np.sort(row)
            out[t] = 0.5 * (np.float64(s[r_lo]) + np.float64(s[r_hi]))
            center = np.int64(s[r_lo])
    return out


@njit(inline="always")
def _counts(row, v):
    lt = 0
    le = 0
    for c in range(row.size):
        lt += row[c] < v
        le += row[c] <= v
    return lt, le


@njit(inline="always")
def _sorted_median(row, scratch, r_lo, r_hi):
    scratch[:] = row
    scratch.sort()  # NaN sorts last
    if np.isnan(scratch[-1]):
        return scratch[-1]
    return np.float32(0.5) * (scratch[r_lo] + scratch[r_hi])


@njit(inline="always")
def _median_of_row(row, buf, scratch):
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
    # Invariant: count(< a) == ca <= r_lo and count(<= b) == cb >= r_hi + 1.
    a = vmin
    ca = 0
    b = vmax
    cb = C
    p = total / C
    target = 0.5 * (r_lo + r_hi + 1)
    for _ in range(_MAX_PASSES):
        if cb - ca <= _FINAL_WIDTH:
            break
        lt, le = _counts(row, p)
        if lt <= r_lo and le >= r_hi + 1:
            return p  # p is a value holding both middle ranks
        if lt > r_lo and le < r_hi + 1:
            # p sits strictly between the two middle values, so they are its nearest neighbours
            below = vmin
            above = vmax
            for c in range(C):
                v = row[c]
                below = max(below, v if v < p else vmin)
                above = min(above, v if v > p else vmax)
            return np.float32(0.5) * (below + above)
        if lt <= r_lo:
            a = p
            ca = lt
        else:
            b = p
            cb = le
        q = a + (target - ca) / max(cb - ca, 1) * (b - a)
        if not (a < q < b):
            q = 0.5 * (a + b)
        p = np.float32(q)
    if cb - ca > _FINAL_WIDTH:
        return _sorted_median(row, scratch, r_lo, r_hi)
    n = 0
    for c in range(C):  # branchless gather of the values in [a, b]
        v = row[c]
        buf[n] = v
        n += (v >= a) & (v <= b)
    middle = buf[:n]
    middle.sort()
    return np.float32(0.5) * (middle[r_lo - ca] + middle[r_hi - ca])


# Reassociation only reorders the sum that seeds the search, so the result stays exact.
@njit(nogil=True, cache=True, fastmath={"reassoc"})
def float_row_median(x):
    T, C = x.shape
    out = np.empty(T, dtype=np.float32)
    buf = np.empty(C + 1, dtype=np.float32)
    scratch = np.empty(C, dtype=np.float32)
    for t in range(T):
        out[t] = _median_of_row(x[t], buf, scratch)
    return out
