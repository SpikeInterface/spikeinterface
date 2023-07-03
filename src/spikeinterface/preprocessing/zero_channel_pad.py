from typing import Union

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class


class TracePaddedRecording(BasePreprocessor):
    """
    A Pre-processor class to lazily pad recordings.

    The class retrieves traces from the parent recording segment and pads them with zeros at both ends.

    Parameters
    ----------
    parent_recording_segment : BaseRecording
        The parent recording segment from which the traces are to be retrieved.
    padding_start : int
        The amount of padding to add to the left of the traces. Default is 0. It has to be non-negative
    padding_end : int
        The amount of padding to add to the right of the traces. Default is 0. It has to be non-negative
    fill_value: float
        The value to pad with. Default is 0.
    """

    def __init__(
        self, parent_recording: BaseRecording, padding_start: int = 0, padding_end: int = 0, fill_value: float = 0.0
    ):
        assert padding_end >= 0 and padding_start >= 0, "Paddings must be >= 0"
        super().__init__(recording=parent_recording)

        self.padding_start = padding_start
        self.padding_end = padding_end
        self.fill_value = fill_value
        for segment in parent_recording._recording_segments:
            recording_segment = TracePaddedRecordingSegment(
                segment,
                parent_recording.get_num_channels(),
                self.dtype,
                self.padding_start,
                self.padding_end,
                self.fill_value,
            )
            self.add_recording_segment(recording_segment)

        self._kwargs = dict(
            parent_recording=parent_recording,
            padding_start=padding_start,
            padding_end=padding_end,
            fill_value=fill_value,
        )


class TracePaddedRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment: BaseRecordingSegment,
        num_channels,
        dtype,
        paddign_left,
        padding_end,
        fill_value,
    ):
        self.padding_start = paddign_left
        self.padding_end = padding_end
        self.fill_value = fill_value
        self.num_channels = num_channels
        self.num_samples_in_original_segment = parent_recording_segment.get_num_samples()
        self.dtype = dtype

        super().__init__(parent_recording_segment=parent_recording_segment)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        # This contains the padded elements by default and we add the original traces if necessary
        trace_size = end_frame - start_frame
        if isinstance(channel_indices, (np.ndarray, list)):
            num_channels = len(channel_indices)
        elif channel_indices == slice(None):
            num_channels = self.num_channels
        else:
            raise ValueError(f"Unsupported channel_indices type: {type(channel_indices)} raise an issue on github ")

        # This avoids an extra memory allocation if we are within the confines of the old traces
        if start_frame > self.padding_start and end_frame < self.num_samples_in_original_segment + self.padding_start:
            return self.get_original_traces_shifted(start_frame, end_frame, channel_indices)

        # Else, we start with the full padded traces and allocate the original traces in the middle
        output_traces = np.full(shape=(trace_size, num_channels), fill_value=self.fill_value, dtype=self.dtype)

        # After the padding, the original traces are placed in the middle until the end of the original traces
        if end_frame >= self.padding_start:
            original_traces = self.get_original_traces_shifted(
                start_frame=start_frame,
                end_frame=end_frame,
                channel_indices=channel_indices,
            )

            padded_frames_at_the_start = max(self.padding_start - start_frame, 0)
            end_of_original_traces = padded_frames_at_the_start + original_traces.shape[0]
            output_traces[padded_frames_at_the_start:end_of_original_traces, :] = original_traces

        return output_traces

    def get_original_traces_shifted(self, start_frame, end_frame, channel_indices):
        """
        Retrieves the corresponding original (unpadded) traces from the parent recording segment.

        This method calculates the start and end frames of the original traces, taking into account
        the start padding. It then fetches the original traces using these frames from the parent
        recording segment.

        """
        original_start_frame = max(start_frame - self.padding_start, 0)
        original_end_frame = min(end_frame - self.padding_start, self.num_samples_in_original_segment)
        original_traces = self.parent_recording_segment.get_traces(
            start_frame=original_start_frame,
            end_frame=original_end_frame,
            channel_indices=channel_indices,
        )

        return original_traces

    def get_num_samples(self):
        "Overide the parent method to return the padded number of samples"
        return self.num_samples_in_original_segment + self.padding_start + self.padding_end


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
pad_traces = define_function_from_class(source_class=TracePaddedRecording, name="pad_traces")
