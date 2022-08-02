import numpy as np

from spikeinterface.core.channelslice import ChannelSliceRecording
from spikeinterface.core.core_tools import define_function_from_class

from .basepreprocessor import BasePreprocessor

from ..core import get_random_data_chunks


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
    **random_chunk_kwargs

    Returns
    -------
    remove_bad_channels_recording: RemoveBadChannelsRecording
        The recording extractor without bad channels
    """
    name = 'remove_bad_channels'

    def __init__(self, recording, bad_threshold=5, **random_chunk_kwargs):
        random_data = get_random_data_chunks(recording, **random_chunk_kwargs)

        stds = np.std(random_data, axis=0)
        thresh = bad_threshold * np.median(stds)
        keep_inds, = np.nonzero(stds < thresh)

        parents_chan_ids = recording.get_channel_ids()
        channel_ids = parents_chan_ids[keep_inds]
        self._parent_channel_indices = recording.ids_to_indices(channel_ids)

        BasePreprocessor.__init__(self, recording)
        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_ids)

        self._kwargs = dict(recording=recording.to_dict(), bad_threshold=bad_threshold)
        self._kwargs.update(random_chunk_kwargs)


# function for API
remove_bad_channels = define_function_from_class(source_class=RemoveBadChannelsRecording, name="remove_bad_channels")
