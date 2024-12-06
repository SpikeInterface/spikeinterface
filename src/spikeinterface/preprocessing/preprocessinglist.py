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
from .clip import BlankSaturationRecording, blank_staturation, ClipRecording, clip
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

from .motion import correct_motion

pp_name_to_function = {
    # filter stuff
    "filter": filter,
    "bandpass_filter": bandpass_filter,
    "notch_filter": notch_filter,
    "highpass_filter": highpass_filter,
    "gaussian_filter": gaussian_filter,
    # gain offset stuff
    "normalize_by_quantile": normalize_by_quantile,
    "scale": scale,
    "zscore": zscore,
    "center": center,
    # decorrelation stuff
    "whiten": whiten,
    # re-reference
    "common_reference": common_reference,
    "phase_shift": phase_shift,
    # misc
    "rectify": rectify,
    "clip": clip,
    "blank_staturation": blank_staturation,
    "silence_periods": silence_periods,
    "remove_artifacts": remove_artifacts,
    "zero_channel_pad": zero_channel_pad,
    "deepinterpolate": deepinterpolate,
    "resample": resample,
    "decimate": decimate,
    "highpass_spatial_filter": highpass_spatial_filter,
    "interpolate_bad_channels": interpolate_bad_channels,
    "depth_order": depth_order,
    "average_across_direction": average_across_direction,
    "directional_derivative": directional_derivative,
    "astype": astype,
    "unsigned_to_signed": unsigned_to_signed,
    "unsigned_to_signed": unsigned_to_signed,
    # motion correction
    "correct_motion": correct_motion,
}

pp_function_to_class = {
    # filter stuff
    "filter": FilterRecording,
    "bandpass_filter": BandpassFilterRecording,
    "notch_filter": NotchFilterRecording,
    "highpass_filter": HighpassFilterRecording,
    "gaussian_filter": GaussianFilterRecording,
    # gain offset stuff
    "normalize_by_quantile": NormalizeByQuantileRecording,
    "scale": ScaleRecording,
    "zscore": ZScoreRecording,
    "center": CenterRecording,
    # decorrelation stuff
    "whiten": WhitenRecording,
    # re-reference
    "common_reference": CommonReferenceRecording,
    "phase_shift": PhaseShiftRecording,
    # misc
    "rectify": RectifyRecording,
    "clip": ClipRecording,
    "blank_staturation": BlankSaturationRecording,
    "silence_periods": SilencedPeriodsRecording,
    "remove_artifacts": RemoveArtifactsRecording,
    "zero_channel_pad": ZeroChannelPaddedRecording,
    "deepinterpolate": DeepInterpolatedRecording,
    "resample": ResampleRecording,
    "decimate": DecimateRecording,
    "highpass_spatial_filter": HighpassSpatialFilterRecording,
    "interpolate_bad_channels": InterpolateBadChannelsRecording,
    "depth_order": DepthOrderRecording,
    "average_across_direction": AverageAcrossDirectionRecording,
    "directional_derivative": DirectionalDerivativeRecording,
    "astype": AstypeRecording,
    "unsigned_to_signed": UnsignedToSignedRecording,
}


preprocessers_full_list = pp_function_to_class.values()

preprocesser_dict = {pp_class.__name__: pp_class for pp_class in preprocessers_full_list}
