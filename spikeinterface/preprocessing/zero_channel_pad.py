from typing import Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class ZeroChannelPaddedRecording(BaseRecording):
    name = 'zero_channel_pad'
    installed = True

    def __init__(self, parent_recording: BaseRecording,
                 num_channels: int, channel_mapping: Union[list, None] = None):
        """Pads a recording with channels that contain only zero.

        Parameters
        ----------
        parent_recording : BaseRecording
            recording to zero-pad
        num_channels : int
            Total number of channels in the zero-channel-padded recording
        channel_mapping : Union[list, None], optional
            Mapping from the channel index in the original recording to the zero-channel-padded recording, 
            by default None.
            If None, sorts the channel indices in ascending y channel location and puts them at the 
            beginning of the zero-channel-padded recording.
        """
        BaseRecording.__init__(self, parent_recording.get_sampling_frequency(), 
                               np.arange(num_channels), parent_recording.get_dtype())
        
        if channel_mapping is not None:
            assert len(channel_mapping) == parent_recording.get_num_channels(), \
                "The new mapping must be specified for all channels."
            assert max(channel_mapping) < num_channels, ("The new mapping cannot exceed total number of channels "
                                                         "in the zero-chanenl-padded recording.")
        else:
            if 'locations' in parent_recording.get_property_keys() or \
                'contact_vector' in parent_recording.get_property_keys():
                self.channel_mapping = np.argsort(parent_recording.get_channel_locations()[:, 1])
            else:
                self.channel_mapping = np.arange(parent_recording.get_num_channels())

        self.parent_recording = parent_recording
        self.num_channels = num_channels
        for segment in parent_recording._recording_segments:
            recording_segment = ZeroChannelPaddedRecordingSegment(
                segment, self.num_channels, self.channel_mapping)
            self.add_recording_segment(recording_segment)
        
        # only copy relevant metadata and properties
        parent_recording.copy_metadata(self, only_main=True)
        prop_keys = parent_recording.get_property_keys()

        for k in prop_keys:
            values = self.get_property(k)
            if values is not None:
                self.set_property(k, values, ids=self.channel_ids[self.channel_mapping])

        self._kwargs = dict(recording=parent_recording.to_dict(),
                            num_channels=num_channels, channel_mapping=channel_mapping)


class ZeroChannelPaddedRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment: BaseRecordingSegment, num_channels: int,
                 channel_mapping: list):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.num_channels = num_channels
        self.channel_mapping = channel_mapping

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        traces = np.zeros((end_frame-start_frame, self.num_channels))
        traces[:, self.channel_mapping] = self.parent_recording_segment.get_traces(start_frame=start_frame,
                                                                                   end_frame=end_frame,
                                                                                   channel_indices=self.channel_mapping)
        return traces[:, channel_indices]


# function for API
zero_channel_pad = define_function_from_class(source_class=ZeroChannelPaddedRecording, name="zero_channel_pad")
