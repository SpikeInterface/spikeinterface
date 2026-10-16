"""numba kernel for ``_row_median``, kept at module level so that ``cache=True`` can find it again.

``int_row_median`` returns exactly what ``np.median(x, axis=1)`` returns for integer ``x``. For a row of ``C``
values, sorted, the median is the mean of the values at ranks ``r_lo = (C - 1) // 2`` and ``r_hi = C // 2``,
which are the same rank when ``C`` is odd.

The kernel counts each row into a histogram instead of sorting it. Filtered integer traces take few distinct
values, and the median moves little from one sample to the next, so a window of bins centred on the previous
row's median almost always holds both middle values. A cumulative sum over the bins then gives them directly.
Rows whose middle values fall outside the window are sorted in full, so every path returns the exact median.
"""

import numpy as np
from numba import njit


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
