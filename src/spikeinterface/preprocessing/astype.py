from __future__ import annotations

import numpy as np

from ..core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from .filter import fix_dtype


class AstypeRecording(BasePreprocessor):
    """The spikeinterface analog of numpy.astype

    Converts a recording to another dtype on the fly.

    For recording with an unsigned dtype, please use the `unsigned_to_signed` preprocessing function.

    Parameters
    ----------
    dtype : None | str | dtype, default: None
        dtype of the output recording. If None, takes dtype from input `recording`.
    recording : Recording
        The recording extractor to be converted.
    round : Bool | None, default: None
        If True, will round the values to the nearest integer using `numpy.round`.
        If None and dtype is an integer, will round floats to nearest integer.

    Returns
    -------
    astype_recording : AstypeRecording
        The converted recording extractor object
    """

    name = "astype"

    def __init__(
        self,
        recording,
        dtype=None,
        round: bool | None = None,
    ):
        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        if round is None:
            round = np.issubdtype(dtype, np.integer)

        for parent_segment in recording._recording_segments:
            rec_segment = AstypeRecordingSegment(
                parent_segment,
                dtype,
                round,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            dtype=dtype_.str,
            round=round,
        )


class AstypeRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        dtype,
        round: bool,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.dtype = dtype
        self.round = round

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        if self.round:
            np.round(traces, out=traces)
        return traces.astype(self.dtype, copy=False)


# function for API
astype = define_function_from_class(source_class=AstypeRecording, name="astype")
