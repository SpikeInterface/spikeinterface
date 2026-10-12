"""Exact per-row median of a ``(T, C)`` array, for common median referencing.

``np.median(axis=1)`` partitions each row with data-dependent branches, mispredicting about one per value.
Integer rows are counted into a histogram over a window around the previous row's median, since filtered
integer traces repeat few values and adjacent samples have nearly the same median. float32 rows narrow a value
interval around the two middle ranks with branchless counting passes that vectorise over the row, seeded at the
row mean and stepped by interpolating the row's empirical CDF, then sort the few values left in the interval.
Both equal ``np.median``, including for float rows holding NaN or inf.
"""

from __future__ import annotations

import importlib.util
from functools import cache

import numpy as np

HAVE_NUMBA = importlib.util.find_spec("numba") is not None

# Histogram bins; a row whose middle values fall outside the window is sorted instead.
_INT_WINDOW = 1024
# Below these channel counts the kernels' per-row setup costs more than np.median's partition.
_MIN_INT_CHANNELS = 4
_MIN_FLOAT_CHANNELS = 64


def row_median(traces: np.ndarray) -> np.ndarray:
    """Median of each row of a 2D array, as ``(T, 1)`` in ``np.median``'s output dtype.

    Integer input of at most 16 bits and at least 4 channels, and float32 input of at least 64 channels, use numba
    kernels when numba is installed; other input uses ``np.median``.
    """
    if HAVE_NUMBA and traces.ndim == 2 and traces.shape[0] > 0 and traces.shape[1] > 0:
        if traces.dtype == np.float32 and traces.shape[1] >= _MIN_FLOAT_CHANNELS:
            return _get_float_kernel()(np.ascontiguousarray(traces))[:, np.newaxis]
        if traces.dtype.kind in "iu" and traces.dtype.itemsize <= 2 and traces.shape[1] >= _MIN_INT_CHANNELS:
            # Reading column-major traces in place beats copying them; the float32 kernel is the other way round.
            return _get_int_kernel()(traces, _INT_WINDOW)[:, np.newaxis]
    return np.median(traces, axis=1, keepdims=True)


@cache
def _get_int_kernel():
    from ._row_median_numba import int_row_median

    return int_row_median


@cache
def _get_float_kernel():
    from ._row_median_numba import float_row_median

    return float_row_median
