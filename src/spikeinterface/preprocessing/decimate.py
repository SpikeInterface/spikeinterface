from __future__ import annotations

import numpy as np
from spikeinterface.core.core_tools import (
    define_function_handling_dict_from_class,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from spikeinterface.core import BaseRecordingSegment


class DecimateRecording(BasePreprocessor):
    """
    Decimate the recording extractor traces using array slicing

    Important: This uses simple array slicing for decimation rather than eg scipy.decimate.
    This might introduce aliasing, or skip across signal of interest.
    Consider  spikeinterface.preprocessing.ResampleRecording for safe resampling.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be decimated. Each segment is decimated independently.
    decimation_factor : int
        Step between successive frames sampled from the parent recording.
        The same decimation factor is applied to all segments from the parent recording.
    decimation_offset : int, default: 0
        Index of first frame sampled from the parent recording.
        Expecting `decimation_offset` < `decimation_factor`, and `decimation_offset` < parent_recording.get_num_samples()
        to ensure that the decimated recording has at least one frame. Consider combining DecimateRecording
        with FrameSliceRecording for fine control on the recording start and end frames.
        The same decimation offset is applied to all segments from the parent recording.

    Returns
    -------
    decimate_recording: DecimateRecording
        The decimated recording extractor object. The full traces of the child recording segment
        correspond to the traces of the parent segment as follows:
            ```<decimated_traces> = <parent_traces>[<decimation_offset>::<decimation_factor>]```

    """

    def __init__(
        self,
        recording,
        decimation_factor,
        decimation_offset=0,
    ):
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        if not isinstance(decimation_factor, int) or decimation_factor <= 0:
            raise ValueError(f"Expecting strictly positive integer for `decimation_factor` arg")
        self._decimation_factor = decimation_factor
        if not isinstance(decimation_offset, int) or decimation_factor < 0:
            raise ValueError(f"Expecting positive integer for `decimation_factor` arg")
        parent_min_n_samp = min(
            [recording.get_num_samples(segment_index) for segment_index in range(recording.get_num_segments())]
        )
        if decimation_offset >= decimation_factor or decimation_offset >= parent_min_n_samp:
            raise ValueError(
                f"Expecting `decimation_offset` < `decimation_factor` and `decimation_offset` < parent_segment.get_num_samples() for all segments. "
                f"Consider combining DecimateRecording with FrameSliceRecording for fine control on the recording start/end frames."
            )
        self._decimation_offset = decimation_offset
        decimated_sampling_frequency = self._orig_samp_freq / self._decimation_factor

        BasePreprocessor.__init__(self, recording, sampling_frequency=decimated_sampling_frequency)

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                DecimateRecordingSegment(
                    parent_segment,
                    decimated_sampling_frequency,
                    self._orig_samp_freq,
                    decimation_factor,
                    decimation_offset,
                    self._dtype,
                )
            )

        self._kwargs = dict(
            recording=recording,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
        )


class DecimateRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        parent_recording_segment,
        decimated_sampling_frequency,
        parent_rate,
        decimation_factor,
        decimation_offset,
        dtype,
    ):
        if parent_recording_segment.time_vector is not None:
            time_vector = parent_recording_segment.time_vector[decimation_offset::decimation_factor]
            decimated_sampling_frequency = None
            t_start = None
        else:
            time_vector = None
            if parent_recording_segment.t_start is None:
                t_start = None
            else:
                t_start = parent_recording_segment.t_start + (decimation_offset / parent_rate)

        # Do not use BasePreprocessorSegment bcause we have to reset the sampling rate!
        BaseRecordingSegment.__init__(
            self, sampling_frequency=decimated_sampling_frequency, t_start=t_start, time_vector=time_vector
        )
        self._parent_segment = parent_recording_segment
        self._decimation_factor = decimation_factor
        self._decimation_offset = decimation_offset
        self._dtype = dtype

    def get_num_samples(self):
        parent_n_samp = self._parent_segment.get_num_samples()
        assert self._decimation_offset < parent_n_samp  # Sanity check (already enforced). Formula changes otherwise
        return int(np.ceil((parent_n_samp - self._decimation_offset) / self._decimation_factor))

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Account for offset and end when querying parent traces
        parent_start_frame = self._decimation_offset + start_frame * self._decimation_factor
        parent_end_frame = parent_start_frame + (end_frame - start_frame) * self._decimation_factor

        # And now we can decimate without offsetting
        return self._parent_segment.get_traces(
            parent_start_frame,
            parent_end_frame,
            channel_indices,
        )[
            :: self._decimation_factor
        ].astype(self._dtype)


decimate = define_function_handling_dict_from_class(source_class=DecimateRecording, name="decimate")
