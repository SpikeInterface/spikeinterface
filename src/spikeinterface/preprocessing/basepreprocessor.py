from __future__ import annotations

from spikeinterface.core import BaseRecording, BaseRecordingSegment


class BasePreprocessor(BaseRecording):
    installation_mesg = ""  # err

    def __init__(self, recording, sampling_frequency=None, channel_ids=None, dtype=None):
        assert isinstance(recording, BaseRecording), "'recording' must be a RecordingExtractor"

        if sampling_frequency is None:
            sampling_frequency = recording.get_sampling_frequency()
        if channel_ids is None:
            _channel_ids = recording.channel_ids
        else:
            _channel_ids = channel_ids

        if dtype is None:
            dtype = recording.get_dtype()

        BaseRecording.__init__(self, sampling_frequency, _channel_ids, dtype)
        recording.copy_metadata(self, only_main=False, ids=channel_ids)
        self._parent = recording

        # self._kwargs have to be handled in subclass


class BasePreprocessorSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        raise NotImplementedError
