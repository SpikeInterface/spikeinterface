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

from .scale import scale_to_uV

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
from .interpolate_bad_channels import InterpolateBadChannelsRecording, interpolate_bad_channels
from .average_across_direction import AverageAcrossDirectionRecording, average_across_direction
from .directional_derivative import DirectionalDerivativeRecording, directional_derivative
from .depth_order import DepthOrderRecording, depth_order
from .astype import AstypeRecording, astype
from .unsigned_to_signed import UnsignedToSignedRecording, unsigned_to_signed


preprocessers_full_list = [
    # filter stuff
    FilterRecording,
    BandpassFilterRecording,
    HighpassFilterRecording,
    NotchFilterRecording,
    GaussianFilterRecording,
    # gain offset stuff
    NormalizeByQuantileRecording,
    ScaleRecording,
    CenterRecording,
    ZScoreRecording,
    # decorrelation stuff
    WhitenRecording,
    # re-reference
    CommonReferenceRecording,
    PhaseShiftRecording,
    # misc
    RectifyRecording,
    ClipRecording,
    BlankSaturationRecording,
    SilencedPeriodsRecording,
    RemoveArtifactsRecording,
    ZeroChannelPaddedRecording,
    DeepInterpolatedRecording,
    ResampleRecording,
    DecimateRecording,
    HighpassSpatialFilterRecording,
    InterpolateBadChannelsRecording,
    DepthOrderRecording,
    AverageAcrossDirectionRecording,
    DirectionalDerivativeRecording,
    AstypeRecording,
    UnsignedToSignedRecording,
]

preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
