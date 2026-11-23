import functools

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.job_tools import get_per_step_job_kwargs, get_step_chunk_size


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


def split_requests_by_per_step_job_kwargs(get_traces):
    """
    Wrap the `get_traces` of a preprocessing segment so that, inside a job with `per_step_job_kwargs`, requests longer
    than the step's chunk size are computed in pieces and assembled.
    """

    @functools.wraps(get_traces)
    def get_traces_in_pieces(self, start_frame=None, end_frame=None, channel_indices=None):
        per_step_job_kwargs = get_per_step_job_kwargs()
        if not per_step_job_kwargs:
            return get_traces(self, start_frame, end_frame, channel_indices)

        from .preprocessing_classes import _all_preprocesser_dict

        recording = self.parent_extractor
        function = _all_preprocesser_dict.get(type(recording))
        options = per_step_job_kwargs.get(function.__name__) if function is not None else None
        if options is None:
            return get_traces(self, start_frame, end_frame, channel_indices)

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        # Pieces of equal length instead of full pieces and a remainder. A request a bit longer than the
        # chunk size, for example one that includes the margin of the step below, would otherwise leave
        # a tiny last piece that the step computes with its own full margin.
        chunk_size = get_step_chunk_size(options, recording)
        num_samples = end_frame - start_frame
        num_pieces = -(-num_samples // chunk_size)
        if num_pieces <= 1:
            return get_traces(self, start_frame, end_frame, channel_indices)

        piece_boundaries = start_frame + np.arange(num_pieces + 1) * num_samples // num_pieces
        traces = None
        for piece_start, piece_end in zip(piece_boundaries[:-1], piece_boundaries[1:]):
            piece = get_traces(self, int(piece_start), int(piece_end), channel_indices)
            if traces is None:
                traces = np.empty((num_samples, piece.shape[1]), dtype=piece.dtype)
            traces[piece_start - start_frame : piece_end - start_frame] = piece
        return traces

    return get_traces_in_pieces


class BasePreprocessorSegment(BaseRecordingSegment):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "get_traces" in cls.__dict__:
            cls.get_traces = split_requests_by_per_step_job_kwargs(cls.__dict__["get_traces"])

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
