from .filter import (FilterRecording, filter,
                     BandpassFilterRecording, bandpass_filter,
                     NotchFilterRecording, notch_filter,
                     )
from .normalize_scale import (
    NormalizeByQuantileRecording, normalize_by_quantile,
    ScaleRecording, scale,
    CenterRecording, center)
from .whiten import WhitenRecording, whiten
from .rectify import RectifyRecording, rectify
from .clip import (
    BlankSaturationRecording, blank_staturation,
    ClipRecording, clip)
from .common_reference import CommonReferenceRecording, common_reference
from .remove_artifacts import RemoveArtifactsRecording, remove_artifacts
from .remove_bad_channels import RemoveBadChannelsRecording, remove_bad_channels

preprocessers_full_list = [
    # filter stuff
    FilterRecording,
    BandpassFilterRecording,
    NotchFilterRecording,

    # gain offset stuff
    NormalizeByQuantileRecording,
    ScaleRecording,
    CenterRecording,

    # decorrelation stuff
    WhitenRecording,

    # re-reference
    CommonReferenceRecording,

    # misc
    RectifyRecording,
    BlankSaturationRecording,
    RemoveArtifactsRecording,
    RemoveBadChannelsRecording,

    # TODO: @alessio this one  is for you
    # ResampleRecording,

]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
