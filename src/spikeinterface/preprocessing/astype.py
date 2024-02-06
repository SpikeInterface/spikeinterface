from __future__ import annotations

from ..core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from .filter import fix_dtype


class AstypeRecording(BasePreprocessor):
    """The spikeinterface analog of numpy.astype

    Converts a recording to another dtype on the fly.

    For recording with an unsigned dtype, please use the `unsigned_to_signed` preprocessing function.
    """

    name = "astype"

    def __init__(
        self,
        recording,
        dtype=None,
    ):
        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = AstypeRecordingSegment(
                parent_segment,
                dtype,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            dtype=dtype_.str,
        )


class AstypeRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        return traces.astype(self.dtype, copy=False)


# function for API
astype = define_function_from_class(source_class=AstypeRecording, name="astype")
