from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.recording_tools import get_rec_attributes, do_recording_attributes_match


class AddRecordings(BaseRecording):
    def __init__(self, recording1, recording2):
        assert all(
            isinstance(rec, BaseRecording) for rec in [recording1, recording2]
        ), "'recordings' must be a list of RecordingExtractor"

        rec_attrs2 = get_rec_attributes(recording2)
        assert do_recording_attributes_match(
            recording1, rec_attrs2
        ), "Both recordings must have the same sampling frequency and channel ids"

        channel_ids = recording1.channel_ids
        sampling_frequency = recording1.sampling_frequency
        dtype = recording1.get_dtype()

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        for segment1, segment2 in zip(recording1._recording_segments, recording2._recording_segments):
            add_segment = AddRecordingSegment(segment1, segment2)
            self.add_recording_segment(add_segment)

        self._kwargs = dict(recording1=recording1, recording2=recording2)


class AddRecordingSegment(BaseRecordingSegment):
    def __init__(self, segment1, segment2):
        BaseRecordingSegment.__init__(self, **segment1.get_times_kwargs())
        self.segment1 = segment1
        self.segment2 = segment2

    def get_num_samples(self):
        return self.segment1.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces1 = self.segment1.get_traces(start_frame, end_frame, channel_indices)
        traces2 = self.segment2.get_traces(start_frame, end_frame, channel_indices)
        return traces1 + traces2


class SubtractRecordings(BaseRecording):
    def __init__(self, recording1, recording2):
        assert all(
            isinstance(rec, BaseRecording) for rec in [recording1, recording2]
        ), "'recordings' must be a list of RecordingExtractor"

        rec_attrs2 = get_rec_attributes(recording2)
        assert do_recording_attributes_match(
            recording1, rec_attrs2
        ), "Both recordings must have the same sampling frequency and channel ids"

        channel_ids = recording1.channel_ids
        sampling_frequency = recording1.sampling_frequency
        dtype = recording1.get_dtype()

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        for segment1, segment2 in zip(recording1._recording_segments, recording2._recording_segments):
            sub_segment = SubtractRecordingSegment(segment1, segment2)
            self.add_recording_segment(sub_segment)

        self._kwargs = dict(recording1=recording1, recording2=recording2)


class SubtractRecordingSegment(BaseRecordingSegment):
    def __init__(self, segment1, segment2):
        BaseRecordingSegment.__init__(self, **segment1.get_times_kwargs())
        self.segment1 = segment1
        self.segment2 = segment2

    def get_num_samples(self):
        return self.segment1.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces1 = self.segment1.get_traces(start_frame, end_frame, channel_indices)
        traces2 = self.segment2.get_traces(start_frame, end_frame, channel_indices)
        return traces1 - traces2
