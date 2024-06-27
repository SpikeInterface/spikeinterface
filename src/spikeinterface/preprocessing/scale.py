from __future__ import annotations

import numpy as np

from spikeinterface.core import BaseRecording
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor


class ScaleTouVRecording(BasePreprocessor):
    """
    Scale raw traces to microvolts (µV).

    This preprocessor uses the channel-specific gain and offset information
    stored in the recording extractor to convert the raw traces to µV units.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor to be scaled. The recording extractor must
        have gains and offsets otherwise an error will be raised.

    Raises
    ------
    AssertionError
        If the recording extractor does not have scaleable traces.
    """

    name = "scale_to_uV"

    def __init__(self, recording: BaseRecording):
        # Importing inside to avoid a circular import
        from spikeinterface.preprocessing.normalize_scale import ScaleRecordingSegment

        dtype = np.dtype("float32")
        BasePreprocessor.__init__(self, recording, dtype=dtype)

        if not recording.has_scaleable_traces():
            error_msg = "Recording must have gains and offsets set to be scaled to µV"
            raise RuntimeError(error_msg)

        gain = recording.get_channel_gains()[None, :]
        offset = recording.get_channel_offsets()[None, :]
        for parent_segment in recording._recording_segments:
            rec_segment = ScaleRecordingSegment(parent_segment, gain, offset, self._dtype)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
        )


scale_to_uV = ScaleTouVRecording
