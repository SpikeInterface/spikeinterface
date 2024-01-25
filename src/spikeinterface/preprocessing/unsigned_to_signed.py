import numpy as np

from ..core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from .filter import fix_dtype


class UnsignedToSignedRecording(BasePreprocessor):
    """
    Converts a recording with unsigned traces to a signed one.
    """

    name = "unsigned_to_signed"

    def __init__(
        self,
        recording,
        bit_depth=None,
    ):
        dtype = np.dtype(recording.dtype)
        assert dtype.kind == "u", "Recording is not unsigned!"
        itemsize = dtype.itemsize
        assert itemsize < 8, "Cannot convert uint64 to int64."
        dtype_signed = dtype.name.replace("uint", "int")

        BasePreprocessor.__init__(self, recording, dtype=dtype_signed)

        for parent_segment in recording._recording_segments:
            rec_segment = UnsignedToSignedRecordingSegment(parent_segment, dtype_signed, bit_depth)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
        )


class UnsignedToSignedRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, dtype_signed, bit_depth):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.dtype_signed = dtype_signed
        self.bit_depth = bit_depth

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        # if uint --> take care of offset
        traces_dtype = traces.dtype
        nbits = traces_dtype.itemsize * 8
        signed_dtype = f"int{2 * (traces_dtype.itemsize) * 8}"
        if self.bit_depth is not None:
            offset = 2 ** (self.bit_depth - 1)
        else:
            # offset = 2 ** (nbits - 1)
            offset = 0
        # upcast to int with double itemsize
        traces = traces.astype(signed_dtype, copy=False) - offset
        return traces.astype(self.dtype_signed, copy=False)


# function for API
unsigned_to_signed = define_function_from_class(source_class=UnsignedToSignedRecording, name="unsigned_to_signed")
