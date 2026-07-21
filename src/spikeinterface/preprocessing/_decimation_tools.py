"""
Helpers for splitting a (potentially large) integer decimation factor into several balanced
sub-factors, so that anti-aliased decimation can be applied as multiple stable scipy.signal.decimate
passes. Shared by DecimateRecording and ResampleRecording.
"""

import math
import warnings

import numpy as np

from spikeinterface.core import get_chunk_with_margin

# scipy.signal.decimate uses an order-8 Chebyshev type I IIR filter by default, and its
# documentation recommends decimating in several balanced steps rather than a single step
# for downsampling factors larger than this value.
_MAX_SINGLE_PASS_DECIMATION = 13


def _prime_factors(n):
    """
    Return the prime factors of a positive integer `n` (ascending, with multiplicity).

    Examples
    --------
    >>> _prime_factors(60)
    [2, 2, 3, 5]
    >>> _prime_factors(17)
    [17]
    """
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def _greedy_pack(primes_desc, num_bins):
    """
    Greedily pack `primes_desc` (largest first) into `num_bins` bins, keeping each bin's
    product <= `_MAX_SINGLE_PASS_DECIMATION` and the bins as balanced as possible.

    Returns the list of bin products, or None if some prime cannot be placed (i.e. `num_bins`
    is too small to keep every bin <= the single-pass limit).

    Examples
    --------
    Pack the prime factors of 48 into two balanced bins (6 and 8):

    >>> _greedy_pack([3, 2, 2, 2, 2], 2)
    [6, 8]

    Two bins cannot hold 2 ** 7 = 128 without a bin exceeding the single-pass limit of 13:

    >>> _greedy_pack([2, 2, 2, 2, 2, 2, 2], 2) is None
    True
    """
    bins = [1] * num_bins
    for prime in primes_desc:
        fitting = [i for i in range(num_bins) if bins[i] * prime <= _MAX_SINGLE_PASS_DECIMATION]
        if not fitting:
            return None
        # Place into the smallest fitting bin (ties broken by index, for determinism).
        target = min(fitting, key=lambda i: (bins[i], i))
        bins[target] *= prime
    return bins


def get_balanced_decimation_factors(decimation_factor):
    """
    Split `decimation_factor` into sub-factors, each <= 13, as balanced as possible (so their
    products are close), for stable multi-pass anti-aliased decimation.

    scipy recommends decimating in several balanced steps rather than one large step when the
    factor exceeds 13 (e.g. 48 -> [8, 6] rather than [12, 4]). The product of the returned
    factors always equals `decimation_factor`.

    If `decimation_factor` has a prime factor greater than 13 (e.g. a large prime such as 17),
    no valid split exists and `[decimation_factor]` is returned; it is the caller's
    responsibility to handle this (e.g., warn that a single, potentially unstable, pass will
    be used).
    """
    if decimation_factor <= _MAX_SINGLE_PASS_DECIMATION:
        return [decimation_factor]

    primes = _prime_factors(decimation_factor)
    if max(primes) > _MAX_SINGLE_PASS_DECIMATION:
        # If a prime factor > 13 cannot be split into sub-13 factors...
        return [decimation_factor]

    primes_desc = sorted(primes, reverse=True)
    # Minimum number of passes so that, ideally, each pass decimates by <= 13.
    num_passes = max(1, math.ceil(math.log(decimation_factor) / math.log(_MAX_SINGLE_PASS_DECIMATION)))
    while num_passes <= len(primes_desc):
        bins = _greedy_pack(primes_desc, num_passes)
        if bins is not None:
            return sorted(bins, reverse=True)
        num_passes += 1
    # Fallback: one prime per pass (always valid since every prime is <= 13).
    return primes_desc


def get_antialiased_decimated_traces(
    parent_segment,
    start_frame,
    end_frame,
    channel_indices,
    decimation_factor,
    decimation_factors,
    margin,
    dtype,
    decimation_offset=0,
):
    """
    Fetch a margined chunk from `parent_segment` and decimate it by `decimation_factor`, applied
    as a cascade of the balanced `decimation_factors` passes of ``scipy.signal.decimate``.

    The margin is rounded up to a multiple of the total `decimation_factor` so that
    ``left_margin // decimation_factor`` is exact; combined with scipy's default
    ``zero_phase=True`` (output sample i maps to filtered input sample i * factor), this keeps the
    downsampled traces aligned across chunks (a chunked read matches a full read). Exactly
    ``end_frame - start_frame`` decimated samples are returned.

    Parameters
    ----------
    parent_segment : BaseRecordingSegment
        The parent segment to read (full-rate) traces from.
    start_frame, end_frame : int
        Output (decimated) frame range to return.
    channel_indices : slice | list | np.ndarray | None
        Channels to read, forwarded to the parent segment.
    decimation_factor : int
        The total decimation factor (the product of `decimation_factors`).
    decimation_factors : list[int]
        The per-pass sub-factors (each <= 13), e.g. from `get_balanced_decimation_factors`.
    margin : int
        Margin in parent samples used to limit anti-aliasing filter edge effects. Rounded up
        internally to a multiple of `decimation_factor`.
    dtype : np.dtype | str
        Output dtype. The decimation runs in float32 and the result is cast to `dtype`.
    decimation_offset : int, default: 0
        Index of the first parent frame, applied to the first output sample only.
    """
    from scipy import signal

    q = decimation_factor
    parent_start_frame = decimation_offset + start_frame * q
    parent_end_frame = parent_start_frame + (end_frame - start_frame) * q
    # Round the margin up to a multiple of q so that left_margin // q is exact.
    margin = int(np.ceil(margin / q) * q)
    parent_traces, left_margin, right_margin = get_chunk_with_margin(
        parent_segment,
        parent_start_frame,
        parent_end_frame,
        channel_indices,
        margin,
        add_reflect_padding=True,
        dtype=np.float32,
    )
    decimated_traces = parent_traces
    for sub_q in decimation_factors:
        decimated_traces = signal.decimate(decimated_traces, q=sub_q, axis=0)
    if np.any(np.isnan(decimated_traces)):
        warnings.warn(
            f"`scipy.signal.decimate` produced NaNs while decimating by {q}. "
            f"Consider a different decimation factor."
        )
    start_drop = left_margin // q
    n_out = end_frame - start_frame
    return decimated_traces[start_drop : start_drop + n_out].astype(dtype)
