import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.base import base_period_dtype


class DeAutozeroRecording(BasePreprocessor):

    def __init__(
        self,
        recording,
        az_samples,
        az_periods,
        voltage_cumsum,
        firmwave_version=None,
    ):
        BasePreprocessor.__init__(self, recording)
        num_channels = recording.get_num_channels()

        for parent_segment in recording._recording_segments:
            rec_segment = DeAutozeroRecordingSegment(
                parent_segment, az_samples, az_periods, voltage_cumsum, num_channels
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            az_samples=az_samples,
            az_periods=az_periods,
            voltage_cumsum=voltage_cumsum,
            firmwave_version=firmwave_version,
        )


class DeAutozeroRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        az_samples,
        az_periods,
        voltage_cumsum,
        num_channels,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.az_samples = az_samples
        self.az_periods = az_periods
        self.voltage_cumsum = voltage_cumsum
        self.num_channels = num_channels

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)

        az_sample_in_range = (self.az_samples > start_frame) & (self.az_samples < end_frame)

        indices = np.where(az_sample_in_range)[0]
        index_before = indices[0] - 1
        index_after = indices[-1] + 1

        recording_shift = np.zeros(np.shape(traces))

        for az_index in range(index_before, index_after):

            if az_index == -1:
                continue

            previous_az_event = max(0, self.az_samples[az_index] - start_frame)
            next_az_event = min(self.az_samples[az_index + 1] - start_frame, end_frame - start_frame)

            recording_shift[previous_az_event:next_az_event, :] = self.voltage_cumsum[az_index, channel_indices]

        return traces - recording_shift


deautozero = define_function_handling_dict_from_class(source_class=DeAutozeroRecording, name="deautozero")

# Tools


def get_autozero_periods_sinaps(recording, autozero_channel, period_method="simple"):

    az_event_occured = np.transpose(autozero_channel.get_traces() == 1024)[0]
    az_samples = np.arange(0, len(az_event_occured)).astype("int64")[az_event_occured]

    az_periods = np.zeros(len(az_samples), dtype=base_period_dtype)
    az_periods["start_sample_index"] = az_samples - 3
    az_periods["end_sample_index"] = az_samples + 3

    return az_samples, az_periods


def get_autozero_information(recording, autozero_channel, baseline_estimate_sample_size=10):

    az_samples, az_periods = get_autozero_periods_sinaps(recording, autozero_channel)
    num_channels = recording.get_num_channels()

    voltage_differences = np.zeros((len(az_periods), num_channels), "int64")

    for az_index_index, az_period in enumerate(az_periods):

        az_envelope = recording.get_traces(
            start_frame=az_period["start_sample_index"] - baseline_estimate_sample_size,
            end_frame=az_period["end_sample_index"] + baseline_estimate_sample_size,
        )

        voltage_before = np.median(az_envelope[:baseline_estimate_sample_size, :], axis=0)
        voltage_after = np.median(az_envelope[-baseline_estimate_sample_size:, :], axis=0)
        voltage_difference = voltage_after - voltage_before
        voltage_differences[az_index_index, :] = voltage_difference

    voltage_cumsum = np.cumsum(voltage_differences, axis=0)

    return az_samples, az_periods, voltage_cumsum
