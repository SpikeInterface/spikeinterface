from typing import List, Union
import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class ZeroTracePaddedRecording(BaseRecording):
    def __init__(self, parent_recording: BaseRecording, padding_left=0, padding_right=0):
        BaseRecording.__init__(
            self,
            parent_recording.get_sampling_frequency(),
            parent_recording.get_channel_ids(),
            parent_recording.get_dtype(),
        )
        self.parent_recording = parent_recording
        self.padding_left = padding_left
        self.padding_right = padding_right
        for segment in parent_recording._recording_segments:
            recording_segment = ZeroTracePaddedRecordingSegment(
                segment, parent_recording.get_num_channels(), self.dtype, self.padding_left, self.padding_right
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = dict(parent_recording=parent_recording, padding_left=padding_left, padding_right=padding_right)


class ZeroTracePaddedRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self, parent_recording_segment: BaseRecordingSegment, num_channels, dtype, paddign_left, padding_right
    ):
        self.padding_left = paddign_left
        self.padding_right = padding_right
        self.num_channels = num_channels
        self.dtype = dtype

        super().__init__(parent_recording_segment=parent_recording_segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        trace_size = end_frame - start_frame
        output_traces = np.zeros((trace_size, self.num_channels), dtype=self.dtype)

        # If the stat_frame is shorter than padding left then we add the zeros to till start_frame is reached
        if end_frame >= self.padding_left:
            shifted_start_frame = max(start_frame - self.padding_left, 0)
            shifted_end_frame = end_frame - self.padding_left
            original_traces = self.parent_recording_segment.get_traces(
                start_frame=shifted_start_frame,
                end_frame=shifted_end_frame,
                channel_indices=channel_indices,
            )

            end_of_left_padding_frame = self.padding_left - start_frame
            start_of_right_padding_frame = end_of_left_padding_frame + self.parent_recording_segment.get_num_samples()
            output_traces[end_of_left_padding_frame:start_of_right_padding_frame, :] = original_traces

        return output_traces

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples() + self.padding_left + self.padding_right


class ZeroChannelPaddedRecording(BaseRecording):
    name = "zero_channel_pad"
    installed = True

    def __init__(self, parent_recording: BaseRecording, num_channels: int, channel_mapping: Union[list, None] = None):
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
        BaseRecording.__init__(
            self, parent_recording.get_sampling_frequency(), np.arange(num_channels), parent_recording.get_dtype()
        )

        if channel_mapping is not None:
            assert (
                len(channel_mapping) == parent_recording.get_num_channels()
            ), "The new mapping must be specified for all channels."
            assert max(channel_mapping) < num_channels, (
                "The new mapping cannot exceed total number of channels " "in the zero-chanenl-padded recording."
            )
        else:
            if (
                "locations" in parent_recording.get_property_keys()
                or "contact_vector" in parent_recording.get_property_keys()
            ):
                self.channel_mapping = np.argsort(parent_recording.get_channel_locations()[:, 1])
            else:
                self.channel_mapping = np.arange(parent_recording.get_num_channels())

        self.parent_recording = parent_recording
        self.num_channels = num_channels
        for segment in parent_recording._recording_segments:
            recording_segment = ZeroChannelPaddedRecordingSegment(segment, self.num_channels, self.channel_mapping)
            self.add_recording_segment(recording_segment)

        # only copy relevant metadata and properties
        parent_recording.copy_metadata(self, only_main=True)
        prop_keys = parent_recording.get_property_keys()

        for k in prop_keys:
            values = self.get_property(k)
            if values is not None:
                self.set_property(k, values, ids=self.channel_ids[self.channel_mapping])

        self._kwargs = dict(
            parent_recording=parent_recording, num_channels=num_channels, channel_mapping=channel_mapping
        )


class ZeroChannelPaddedRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment: BaseRecordingSegment, num_channels: int, channel_mapping: list):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.parent_recording_segment = parent_recording_segment
        self.num_channels = num_channels
        self.channel_mapping = channel_mapping

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        traces = np.zeros((end_frame - start_frame, self.num_channels))
        traces[:, self.channel_mapping] = self.parent_recording_segment.get_traces(
            start_frame=start_frame, end_frame=end_frame, channel_indices=self.channel_mapping
        )
        return traces[:, channel_indices]


# function for API
zero_channel_pad = define_function_from_class(source_class=ZeroChannelPaddedRecording, name="zero_channel_pad")
