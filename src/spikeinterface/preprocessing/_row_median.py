"""Exact per-row median of a ``(T, C)`` array, for common median referencing.

``np.median(axis=1)`` partitions each row with data-dependent branches, mispredicting about one per value.
Float rows are sorted instead, since ``np.sort`` vectorises 32- and 64-bit sorts on most current CPUs. Integer
rows of at most 16 bits, whose sorts it vectorises only on some, go through the histogram kernel in
``_row_median_numba``. Both equal ``np.median``, including for float rows holding NaN or inf.
"""

from __future__ import annotations

import importlib.util
from functools import cache

import numpy as np

HAVE_NUMBA = importlib.util.find_spec("numba") is not None

# Histogram bins; a row whose middle values fall outside the window is sorted instead.
_INT_WINDOW = 1024
# Below these channel counts, np.median's partition costs less than the kernel's per-row setup or the sort.
_MIN_INT_CHANNELS = 4
_MIN_FLOAT_CHANNELS = 3


def row_median(traces: np.ndarray) -> np.ndarray:
    """Median of each row of a 2D array, as ``(T, 1)`` in ``np.median``'s output dtype.

    float32 and float64 input of at least 3 channels is sorted. Native-endian integer input of at most 16 bits
    and at least 4 channels uses a numba kernel when numba is installed. Other input uses ``np.median``.
    """
    if traces.ndim == 2 and traces.shape[0] > 0:
        num_channels = traces.shape[1]
        if traces.dtype in (np.float32, np.float64) and num_channels >= _MIN_FLOAT_CHANNELS:
            return _sorted_row_median(traces)
        if (
            HAVE_NUMBA
            and traces.dtype.kind in "iu"
            and traces.dtype.itemsize <= 2
            and traces.dtype.isnative  # numba rejects byte-swapped arrays
            and num_channels >= _MIN_INT_CHANNELS
        ):
            # The kernel reads column-major traces in place, which beats copying them.
            return _get_int_kernel()(traces, _INT_WINDOW)[:, np.newaxis]
    return np.median(traces, axis=1, keepdims=True)


def _sorted_row_median(traces):
    num_channels = traces.shape[1]
    sorted_traces = np.sort(traces, axis=1)  # NaN sorts last
    lo = sorted_traces[:, (num_channels - 1) // 2]
    hi = sorted_traces[:, num_channels // 2]
    median = 0.5 * (lo + hi)  # the scalar stays weak, so the sum and result keep the input dtype
    median[np.isnan(sorted_traces[:, -1])] = np.nan  # np.median returns NaN for a row holding one
    return median[:, np.newaxis]


@cache
def _get_int_kernel():
    from ._row_median_numba import int_row_median

    return int_row_median
