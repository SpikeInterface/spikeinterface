from typing import Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class

class ZeroChannelPaddedRecording(BasePreprocessor):
    name = 'zero_channelpad'

    def __init__(self, parent_recording: BaseRecording,
                 num_chans: int, channel_mapping: Union[list, None] = None):
        """Pads a recording with channels that contain only zero.

        Parameters
        ----------
        parent_recording : BaseRecording
            recording to zero-pad
        num_chans : int
            Total number of channels in the zero-channel-padded recording
        channel_mapping : Union[list, None], optional
            Mapping from the channel index in the original recording to the zero-channel-padded recording, by default None.
            If None, sorts the channel indices in ascending y channel location and puts them at the beginning of the
            zero channel-padded recording.
        """
        BasePreprocessor.__init__(self, parent_recording)
        
        if channel_mapping is not None:
            assert len(channel_mapping) == parent_recording.get_num_channels(), "The new mapping must be specified for all channels."
            assert max(channel_mapping) < num_chans, "The new mapping cannot exceed total number of channels in the zero chanenl-padded recording."
        else:
            self.channel_mapping = np.argsort(parent_recording.get_channel_locations()[:,1])
        
        self.parent_recording = parent_recording
        self.num_chans = num_chans
        for segment in parent_recording._recording_segments:
            recording_segment = ZeroChannelPaddedRecordingSegment(segment, self.num_chans, self.channel_mapping)
            self.add_recording_segment(recording_segment)
        
        self._kwargs = dict(recording=parent_recording.to_dict(),
                            num_chans=num_chans, channel_mapping=channel_mapping)
    
    def get_num_channels(self):
        return self.num_chans
    
    def get_channel_ids(self):
        return np.arange(self.num_chans)
    
class ZeroChannelPaddedRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment: BaseRecordingSegment, num_chans: int,
                 channel_mapping: list):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.num_chans = num_chans
        self.channel_mapping = channel_mapping

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()
    
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = np.zeros((end_frame-start_frame, self.num_chans))
        traces[:, self.channel_mapping] = self.parent_recording_segment.get_traces(start_frame=start_frame,
                                                                                   end_frame=end_frame,
                                                                                   channel_indices=self.channel_mapping)
        return traces[:,channel_indices]
    

# function for API
zero_channelpad = define_function_from_class(source_class=ZeroChannelPaddedRecording, name="zero_channelpad")
