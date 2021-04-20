import numpy as np


from spikeinterface.core.channelslicerecording import (ChannelSliceRecording, 
    ChannelSliceRecordingSegment)
from .basepreprocessor import BasePreprocessor,BasePreprocessorSegment

from ..utils import get_random_data_for_scaling

class RemoveBadChannelsRecording(BasePreprocessor, ChannelSliceRecording):
    """
    Remove bad channels from the recording extractor given a thershold 
    on standard deviation.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    bad_threshold: float
        If automatic is used, the threshold for the standard deviation over which channels are removed
    seconds: float
        If automatic is used, the number of seconds used to compute standard deviations

    Returns
    -------
    remove_bad_channels_recording: RemoveBadChannelsRecording
        The recording extractor without bad channels
    """
    name = 'remove_bad_channels'

    def __init__(self, recording, bad_threshold=5,
            num_chunks_per_segment=50, chunk_size=500, seed=0):
            
        self._kwargs = dict(recording=recording.to_dict(), bad_threshold=bad_threshold,
                num_chunks_per_segment=num_chunks_per_segment, chunk_size=chunk_size, seed=seed)

        random_data = get_random_data_for_scaling(recording, 
                        num_chunks_per_segment=num_chunks_per_segment,
                        chunk_size=chunk_size, seed=seed)
        
        stds = np.std(random_data, axis=0)
        thresh = bad_threshold * np.median(stds)
        keep_inds, = np.nonzero(stds < thresh)
        
        parents_chan_ids = recording.get_channel_ids()
        channel_ids = parents_chan_ids[keep_inds]
        self._parent_channel_indices = recording.ids_to_indices(channel_ids)
        
        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_ids)
        BasePreprocessor.__init__(self, self)


# function for API
def remove_bad_channels(*args, **kwargs):
    return RemoveBadChannelsRecording(*args, **kwargs)
remove_bad_channels.__doc__ = RemoveBadChannelsRecording.__doc__

