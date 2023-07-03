### PREPROCESSORS ###
from .resample import ResampleRecording, resample
from .filter import (
    FilterRecording,
    filter,
    BandpassFilterRecording,
    bandpass_filter,
    NotchFilterRecording,
    notch_filter,
    HighpassFilterRecording,
    highpass_filter,
)
from .filter_gaussian import GaussianBandpassFilterRecording, gaussian_bandpass_filter
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
from .whiten import WhitenRecording, whiten, compute_whitening_matrix
from .rectify import RectifyRecording, rectify
from .clip import BlankSaturationRecording, blank_staturation, ClipRecording, clip
from .common_reference import CommonReferenceRecording, common_reference
from .remove_artifacts import RemoveArtifactsRecording, remove_artifacts
from .silence_periods import SilencedPeriodsRecording, silence_periods
from .phase_shift import PhaseShiftRecording, phase_shift
from .zero_channel_pad import ZeroChannelPaddedRecording, zero_channel_pad
from .deepinterpolation import DeepInterpolatedRecording, deepinterpolate
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
    GaussianBandpassFilterRecording,
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
    HighpassSpatialFilterRecording,
    InterpolateBadChannelsRecording,
    DepthOrderRecording,
    AverageAcrossDirectionRecording,
    DirectionalDerivativeRecording,
    AstypeRecording,
    UnsignedToSignedRecording,
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
