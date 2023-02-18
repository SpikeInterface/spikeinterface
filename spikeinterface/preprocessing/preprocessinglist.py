### PREPROCESSORS ###
from .clip import BlankSaturationRecording, ClipRecording, blank_staturation, clip
from .common_reference import CommonReferenceRecording, common_reference
from .deepinterpolation import DeepInterpolatedRecording, deepinterpolate
from .filter import (
    BandpassFilterRecording,
    FilterRecording,
    HighpassFilterRecording,
    NotchFilterRecording,
    bandpass_filter,
    filter,
    highpass_filter,
    notch_filter,
)
from .highpass_spatial_filter import (
    HighpassSpatialFilterRecording,
    highpass_spatial_filter,
)
from .interpolate_bad_channels import (
    InterpolateBadChannelsRecording,
    interpolate_bad_channels,
)
from .normalize_scale import (
    CenterRecording,
    NormalizeByQuantileRecording,
    ScaleRecording,
    ZScoreRecording,
    center,
    normalize_by_quantile,
    scale,
    zscore,
)
from .phase_shift import PhaseShiftRecording, phase_shift
from .rectify import RectifyRecording, rectify
from .remove_artifacts import RemoveArtifactsRecording, remove_artifacts
from .resample import ResampleRecording, resample
from .whiten import WhitenRecording, whiten
from .zero_channel_pad import ZeroChannelPaddedRecording, zero_channel_pad

preprocessers_full_list = [
    # filter stuff
    FilterRecording,
    BandpassFilterRecording,
    HighpassFilterRecording,
    NotchFilterRecording,
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
    RemoveArtifactsRecording,
    ZeroChannelPaddedRecording,
    DeepInterpolatedRecording,
    ResampleRecording,
    HighpassSpatialFilterRecording,
    InterpolateBadChannelsRecording,
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
