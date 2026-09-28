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

    # Preprocessors never change the frame numbering (no offset, no resampling), so time
    # handling is a pure pass-through to the parent segment. Delegating live (instead of
    # relying on the time_vector/t_start copied into __init__ above) lets any lazy/offset-aware
    # override further up the chain (e.g. FrameSliceRecordingSegment after a frame_slice) keep
    # working without materializing a full time_vector every time this segment is reconstructed.
    def get_times(self, start_frame=None, end_frame=None):
        return self.parent_recording_segment.get_times(start_frame=start_frame, end_frame=end_frame)

    def get_start_time(self):
        return self.parent_recording_segment.get_start_time()

    def get_end_time(self):
        return self.parent_recording_segment.get_end_time()

    def sample_index_to_time(self, sample_ind):
        return self.parent_recording_segment.sample_index_to_time(sample_ind)

    def time_to_sample_index(self, time_s):
        return self.parent_recording_segment.time_to_sample_index(time_s)

    def get_times_kwargs(self):
        return self.parent_recording_segment.get_times_kwargs()
