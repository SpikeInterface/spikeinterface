from __future__ import annotations

### PREPROCESSORS ###
from .resample import ResampleRecording, resample
from .decimate import DecimateRecording, decimate
from .filter import (
    FilterRecording,
    filter,
    BandpassFilterRecording,
    bandpass_filter,
    NotchFilterRecording,
    notch_filter,
    HighpassFilterRecording,
    highpass_filter,
    causal_filter,
)
from .filter_gaussian import GaussianFilterRecording, gaussian_filter
from .normalize_scale import (
    NormalizeByQuantileRecording,
    normalize_by_quantile,
    ScaleRecording,
    scale,
    ZScoreRecording,
    zscore,
    CenterRecording,
    center,
)

from .scale import scale_to_uV, ScaleToPhysicalUnits, scale_to_physical_units

from .whiten import WhitenRecording, whiten, compute_whitening_matrix
from .rectify import RectifyRecording, rectify
from .clip import BlankSaturationRecording, blank_saturation, ClipRecording, clip
from .common_reference import CommonReferenceRecording, common_reference
from .remove_artifacts import RemoveArtifactsRecording, remove_artifacts
from .silence_periods import SilencedPeriodsRecording, silence_periods
from .phase_shift import PhaseShiftRecording, phase_shift
from .zero_channel_pad import ZeroChannelPaddedRecording, zero_channel_pad
from .deepinterpolation import DeepInterpolatedRecording, deepinterpolate, train_deepinterpolation
from .highpass_spatial_filter import HighpassSpatialFilterRecording, highpass_spatial_filter
from .interpolate_bad_channels import (
    DetectAndInterpolateBadChannelsRecording,
    detect_and_interpolate_bad_channels,
    InterpolateBadChannelsRecording,
    interpolate_bad_channels,
)
from .detect_bad_channels import DetectAndRemoveBadChannelsRecording, detect_and_remove_bad_channels
from .average_across_direction import AverageAcrossDirectionRecording, average_across_direction
from .directional_derivative import DirectionalDerivativeRecording, directional_derivative
from .depth_order import DepthOrderRecording, depth_order
from .astype import AstypeRecording, astype
from .unsigned_to_signed import UnsignedToSignedRecording, unsigned_to_signed

_all_preprocesser_dict = {
    # filter stuff
    FilterRecording: filter,
    BandpassFilterRecording: bandpass_filter,
    HighpassFilterRecording: highpass_filter,
    NotchFilterRecording: notch_filter,
    GaussianFilterRecording: gaussian_filter,
    # gain offset stuff
    NormalizeByQuantileRecording: normalize_by_quantile,
    ScaleRecording: scale,
    CenterRecording: center,
    ZScoreRecording: zscore,
    ScaleToPhysicalUnits: scale_to_physical_units,
    # decorrelation stuff
    WhitenRecording: whiten,
    # re-reference
    CommonReferenceRecording: common_reference,
    PhaseShiftRecording: phase_shift,
    # bad channel detection/interpolation
    DetectAndRemoveBadChannelsRecording: detect_and_remove_bad_channels,
    DetectAndInterpolateBadChannelsRecording: detect_and_interpolate_bad_channels,
    # misc
    RectifyRecording: rectify,
    ClipRecording: clip,
    BlankSaturationRecording: blank_saturation,
    SilencedPeriodsRecording: silence_periods,
    RemoveArtifactsRecording: remove_artifacts,
    ZeroChannelPaddedRecording: zero_channel_pad,
    DeepInterpolatedRecording: deepinterpolate,
    ResampleRecording: resample,
    DecimateRecording: decimate,
    HighpassSpatialFilterRecording: highpass_spatial_filter,
    InterpolateBadChannelsRecording: interpolate_bad_channels,
    DepthOrderRecording: depth_order,
    AverageAcrossDirectionRecording: average_across_direction,
    DirectionalDerivativeRecording: directional_derivative,
    AstypeRecording: astype,
    UnsignedToSignedRecording: unsigned_to_signed,
}
# we control import in the preprocessing init by setting an __all__

# pp_function.__name__ gives the name of the function that users should use
__all__ = [pp_function.__name__ for pp_function in _all_preprocesser_dict.values()]
__all__.extend(
    [scale_to_uV.__name__, compute_whitening_matrix.__name__, train_deepinterpolation.__name__, causal_filter.__name__]
)

preprocessor_dict = {pp_class.__name__: pp_function for pp_class, pp_function in _all_preprocesser_dict.items()}
__all__.append("preprocessor_dict")
