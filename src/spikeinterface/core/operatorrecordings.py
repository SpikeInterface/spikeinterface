from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.recording_tools import get_rec_attributes, do_recording_attributes_match


class BaseOperatorRecording(BaseRecording):
    """Base class for operator recordings."""

    def __init__(self, recording1, recording2, operator: str):
        assert operator in ["add", "subtract"], "Operator must be 'add' or 'subtract'"
        assert all(
            isinstance(rec, BaseRecording) for rec in [recording1, recording2]
        ), "'recordings' must be a list of RecordingExtractor"

        rec_attrs2 = get_rec_attributes(recording2)
        assert do_recording_attributes_match(
            recording1, rec_attrs2
        ), "Both recordings must have the same sampling frequency and channel ids"
        assert self.are_times_kwargs_compatible(
            recording1, recording2
        ), "Both recordings must have the same time parameters"

        channel_ids = recording1.channel_ids
        sampling_frequency = recording1.sampling_frequency
        dtype = recording1.get_dtype()

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        for segment1, segment2 in zip(recording1._recording_segments, recording2._recording_segments):
            add_segment = OperatorRecordingSegment(segment1, segment2, operator)
            self.add_recording_segment(add_segment)

        self._kwargs = dict(recording1=recording1, recording2=recording2, operator=operator)

    def are_times_kwargs_compatible(self, recording1, recording2) -> bool:
        import numpy as np

        for segment_index in range(recording1.get_num_segments()):
            time_kwargs1 = recording1._recording_segments[segment_index].get_times_kwargs()
            time_kwargs2 = recording2._recording_segments[segment_index].get_times_kwargs()
            for key in time_kwargs1.keys():
                val1 = time_kwargs1[key]
                val2 = time_kwargs2[key]
                if (val1 is None and val2 is not None) or (val1 is not None and val2 is None):
                    return False
                if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                    if not np.array_equal(val1, val2):
                        return False
                else:
                    if val1 != val2:
                        return False
        return True


class OperatorRecordingSegment(BaseRecordingSegment):
    def __init__(self, segment1, segment2, operator: str):
        BaseRecordingSegment.__init__(self, **segment1.get_times_kwargs())
        self.segment1 = segment1
        self.segment2 = segment2
        self.operator = operator

    def get_num_samples(self):
        return self.segment1.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces1 = self.segment1.get_traces(start_frame, end_frame, channel_indices)
        traces2 = self.segment2.get_traces(start_frame, end_frame, channel_indices)
        if self.operator == "add":
            return traces1 + traces2
        elif self.operator == "subtract":
            return traces1 - traces2
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class AddRecordings(BaseOperatorRecording):
    def __init__(self, recording1, recording2):
        BaseOperatorRecording.__init__(self, recording1, recording2, operator="add")


class SubtractRecordings(BaseOperatorRecording):
    def __init__(self, recording1, recording2):
        BaseOperatorRecording.__init__(self, recording1, recording2, operator="subtract")
