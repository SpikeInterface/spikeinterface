### PREPROCESSORS ###
from .resample import ResampleRecording
from .filter import (FilterRecording, filter,
                     BandpassFilterRecording, bandpass_filter,
                     NotchFilterRecording, notch_filter,
                     HighpassFilterRecording, highpass_filter,
                     )
from .normalize_scale import (
    NormalizeByQuantileRecording, normalize_by_quantile,
    ScaleRecording, scale,
    ZScoreRecording, zscore,
    CenterRecording, center)
from .whiten import WhitenRecording, whiten
from .rectify import RectifyRecording, rectify
from .clip import (
    BlankSaturationRecording, blank_staturation,
    ClipRecording, clip)
from .common_reference import CommonReferenceRecording, common_reference
from .remove_artifacts import RemoveArtifactsRecording, remove_artifacts
from .resample import ResampleRecording, resample
from .phase_shift import PhaseShiftRecording, phase_shift
from .zero_channel_pad import ZeroChannelPaddedRecording, zero_channel_pad
from .deepinterpolation import DeepInterpolatedRecording, deepinterpolate
from .highpass_spatial_filter import HighpassSpatialFilterRecording, highpass_spatial_filter
from .interpolate_bad_channels import InterpolateBadChannelsRecording, interpolate_bad_channels


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
    InterpolateBadChannelsRecording
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
