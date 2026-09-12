"""Rational rate selection and FIR design for polyphase resampling."""

import math
import warnings
from fractions import Fraction

import numpy as np


def get_resampling_factors(parent_rate, resample_rate, max_denominator):
    """Select the closest ratio under the denominator limit and warn about rate differences."""
    if not math.isfinite(parent_rate) or parent_rate <= 0:
        raise ValueError("The parent sampling frequency must be finite and positive")
    if not math.isfinite(resample_rate) or resample_rate <= 0:
        raise ValueError("resample_rate must be finite and positive")
    if not isinstance(max_denominator, int) or max_denominator < 1:
        raise ValueError("max_denominator must be a positive integer")

    ratio = (Fraction(float(resample_rate)) / Fraction(float(parent_rate))).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator
    if up == 0:
        raise ValueError("The requested rate is too low for max_denominator; increase max_denominator")
    achieved_rate = float(Fraction(float(parent_rate)) * ratio)
    if not math.isclose(achieved_rate, resample_rate, rel_tol=1e-12, abs_tol=0):
        error_ppm = (achieved_rate / resample_rate - 1) * 1e6
        warnings.warn(
            f"Requested resample_rate={resample_rate:.16g} Hz; the polyphase ratio {up}/{down} "
            f"achieves {achieved_rate:.16g} Hz ({error_ppm:+.6g} ppm). The output sampling frequency "
            "uses the achieved rate. Increase max_denominator for a closer approximation.",
            stacklevel=2,
        )
    return up, down, achieved_rate


def get_polyphase_filter(sampling_frequency, up, down, margin_ms):
    """Design SciPy's default FIR and cover its support on an aligned input grid."""
    from scipy.signal import firwin

    if margin_ms is not None and (not math.isfinite(margin_ms) or margin_ms < 0):
        raise ValueError("margin_ms must be finite and nonnegative, or None")

    if up == down == 1:
        return np.ones(1), 0

    # Important! The multiplier 10 and Kaiser parameter 5.0 come directly from SciPy’s
    # default resample_poly design. Don't change them!
    half_length = 10 * max(up, down)
    coefficients = firwin(
        2 * half_length + 1,  # odd length gives a symmetric filter with a central sample
        1.0 / max(up, down),
        window=("kaiser", 5.0),
    )

    margin = (half_length + up - 1) // up  # Convert filter support to input samples
    if margin_ms is not None:
        margin = max(margin, math.ceil(margin_ms * sampling_frequency / 1000))
    margin = ((margin + down - 1) // down) * down
    return coefficients, margin
