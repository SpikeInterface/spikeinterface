from .filter import (FilterRecording, filter,
        BandpassFilterRecording,bandpass_filter, 
        NotchFilterRecording, notch_filter,
        )
from .normalize_scale import (
    NormalizeByQuantileRecording, normalize_by_quantile,
    ScaleRecording, scale,
    CenterRecording, center)
from .whiten import WhitenRecording, whiten
from.rectify import RectifyRecording, rectify
from .clip import (
    BlankSaturationRecording, blank_staturation,
    ClipRecording, clip)


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
    #filter stuff
    FilterRecording,
    BandpassFilterRecording,
    NotchFilterRecording,
    
    #gain offset stuff
    NormalizeByQuantileRecording,
    ScaleRecording,
    CenterRecording,
    
    # decorrelation stuff
    WhitenRecording,
    
    # misc
    RectifyRecording,
    BlankSaturationRecording,
    
    
    #~ CommonReferenceRecording,
    #~ ResampleRecording,
    #~ RemoveArtifactsRecording,
    #~ RemoveBadChannelsRecording,
    #~ ClipRecording,

]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.name: pp_class for pp_class in preprocessers_full_list}
