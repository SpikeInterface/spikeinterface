
from .filter import (FilterRecording, filter,
        BandpassFilterRecording,bandpass_filter, 
        NotchFilterRecording, notch_filter,
        )
from .normalize import (
    NormalizeByQuantileRecording, normalize_by_quantile,
    ScaleRecording, scale)

from .whiten import (WhitenRecording, whiten)


#~ from .notch_filter import notch_filter, NotchFilterRecording
#~ from .whiten import whiten, WhitenRecording
#~ from .common_reference import common_reference, CommonReferenceRecording
#~ from .resample import resample, ResampleRecording
#~ from .rectify import rectify, RectifyRecording
#~ from .remove_artifacts import remove_artifacts, RemoveArtifactsRecording
#~ from .transform import transform, TransformRecording
#~ from .remove_bad_channels import remove_bad_channels, RemoveBadChannelsRecording
#~ from .normalize_by_quantile import normalize_by_quantile, NormalizeByQuantileRecording
#~ from .clip import clip, ClipRecording
#~ from .blank_saturation import blank_saturation, BlankSaturationRecording
#~ from .center import center, CenterRecording

preprocessers_full_list = [
    FilterRecording,
    BandpassFilterRecording,
    NotchFilterRecording,
    
    NormalizeByQuantileRecording,
    ScaleRecording,
    
    WhitenRecording,
    
    #~ CommonReferenceRecording,
    #~ ResampleRecording,
    #~ RectifyRecording,
    #~ RemoveArtifactsRecording,
    #~ RemoveBadChannelsRecording,
    #~ TransformRecording,
    #~ ClipRecording,
    #~ BlankSaturationRecording,
    #~ CenterRecording
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
