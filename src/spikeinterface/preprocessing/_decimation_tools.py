"""
Helpers for splitting a (potentially large) integer decimation factor into several balanced
sub-factors, so that anti-aliased decimation can be applied as multiple stable scipy.signal.decimate
passes. Shared by DecimateRecording and ResampleRecording.
"""

import math
import warnings

# scipy.signal.decimate uses an order-8 Chebyshev type I IIR filter by default, and its
# documentation recommends decimating in several steps rather than a single step
# for downsampling factors larger than this value.
_MAX_SINGLE_PASS_DECIMATION = 13


def get_resampling_margin(sampling_frequency, margin_ms, decimation_factors=None):
    """Return the margin in input samples, using an automatic estimate when margin_ms is None.

    The estimate is an extension of the pole-decay heuristic illustrated in SciPy's filtfilt
    documentation (theirs is for a single stage, extended here to the cascade).

    Automatic margins are at least 100 ms and aligned to the total decimation factor.
    FFT resampling margins remain 100 ms because this estimate only applies to IIR decimation.
    An explicit margin_ms overrides the estimate.
    """
    if margin_ms is not None:
        if not math.isfinite(margin_ms) or margin_ms < 0:
            raise ValueError("margin_ms must be finite and nonnegative, or None")
        return int(margin_ms * sampling_frequency / 1000)

    margin = math.ceil(0.1 * sampling_frequency)
    if decimation_factors is None:
        return margin

    from scipy.signal import cheby1

    # Estimation method: for each default Chebyshev IIR stage, 
    # estimate settling as ceil(log(1e-6) / log(r)), where r is the largest pole magnitude. 
    # Convert each stage's estimate to input samples and sum them.
    cascade_margin = 0
    input_stride = 1
    for factor in decimation_factors:
        _, poles, _ = cheby1(8, 0.05, 0.8 / factor, output="zpk")
        radius = max(abs(poles))
        if not 0 < radius < 1:
            raise ValueError("Cannot estimate a stable decimation margin. Specify margin_ms explicitly.")
        cascade_margin += input_stride * math.ceil(math.log(1e-6) / math.log(radius))
        input_stride *= factor

    margin = max(margin, cascade_margin)
    return ((margin + input_stride - 1) // input_stride) * input_stride


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


def _greedy_pack(primes_desc, num_bins, max_factor=_MAX_SINGLE_PASS_DECIMATION):
    """
    Greedily pack `primes_desc` (largest first) into `num_bins` bins, keeping each bin's
    product <= `max_factor` and the bins as balanced as possible.

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
        fitting = [i for i in range(num_bins) if bins[i] * prime <= max_factor]
        if not fitting:
            return None
        # Place into the smallest fitting bin (ties broken by index, for determinism).
        target = min(fitting, key=lambda i: (bins[i], i))
        bins[target] *= prime
    return bins


def get_balanced_decimation_factors(decimation_factor, max_factor=_MAX_SINGLE_PASS_DECIMATION):
    """
    Split `decimation_factor` into balanced sub-factors no greater than `max_factor`.

    SciPy recommends multiple IIR decimation passes for factors above 13.
    Balancing the factors (e.g. 48 -> [8, 6] rather than [12, 4]) further aids stability.
    The product of the returned factors always equals `decimation_factor`.

    If `decimation_factor` has a prime factor greater than 13 (e.g. a large prime such as 17),
    no valid split exists, a warning is issued, and `[decimation_factor]` is returned;
    it is the caller's responsibility to handle this (e.g., warn that a single,
    potentially unstable, pass will be used).
    """
    if not isinstance(max_factor, int) or max_factor < 2:
        raise ValueError("max_factor must be an integer greater than one")
    if decimation_factor <= max_factor:
        return [decimation_factor]

    primes = _prime_factors(decimation_factor)
    if max(primes) > max_factor:
        warnings.warn(
            f"`decimation_factor`={decimation_factor} cannot be split into anti-aliasing passes of <= {max_factor} "
            f"(it has a prime factor > {max_factor}). A single `scipy.signal.decimate` pass will be used, "
            f"which may be unstable. Consider a `decimation_factor` without large prime factors.",
            stacklevel=2,
        )
        return [decimation_factor]

    primes_desc = sorted(primes, reverse=True)
    # Minimum number of passes so that, ideally, each pass decimates by <= max_factor.
    num_passes = max(1, math.ceil(math.log(decimation_factor) / math.log(max_factor)))
    while num_passes <= len(primes_desc):
        bins = _greedy_pack(primes_desc, num_passes, max_factor)
        if bins is not None:
            return sorted(bins, reverse=True)
        num_passes += 1
    # Fallback: one prime per pass (always valid since every prime is <= max_factor).
    return primes_desc
