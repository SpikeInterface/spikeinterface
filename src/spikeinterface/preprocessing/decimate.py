import numpy as np
from spikeinterface.core.core_tools import (
    define_function_from_class,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from ..core import BaseRecordingSegment


class DecimateRecording(BasePreprocessor):
    """
    Decimate the recording extractor traces using array slicing

    Important: This uses simple array slicing for decimation rather than eg scipy.decimate.
    This might introduce aliasing, or skip across signal of interest.
    Consider  spikeinterface.preprocessing.ResampleRecording for safe resampling.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    decimation_frame_step : int
    decimation_frame_start : int, default 0

    Returns
    -------
    decimate_recording: DecimateRecording
        The decimated recording extractor object. The full traces of the child recording correspond
        to the parent traces as follows:
            ```<decimated_traces> = <parent_traces>[<decimation_frame_start>::<decimation_frame_step>]```

    """

    name = "decimate"

    def __init__(
        self,
        recording,
        decimation_frame_step,
        decimation_frame_start=0,
    ):
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        if not isinstance(decimation_frame_step, int) or decimation_frame_step <= 0:
            raise ValueError(f"Expecting strictly positive integer for `decimation_frame_step` arg")
        self._decimation_frame_step = decimation_frame_step
        if not isinstance(decimation_frame_start, int) or decimation_frame_step < 0:
            raise ValueError(f"Expecting positive integer for `decimation_frame_step` arg")
        self._decimation_frame_start = decimation_frame_start
        resample_rate = self._orig_samp_freq / self._decimation_frame_step
        self._sampling_frequency = resample_rate
        # fix_dtype not always returns the str, make sure it does
        dtype = fix_dtype(recording, None).str

        BasePreprocessor.__init__(self, recording, sampling_frequency=resample_rate, dtype=dtype)

        # in case there was a time_vector, it will be dropped for sanity.
        # This is not necessary but consistent with ResampleRecording
        for parent_segment in recording._recording_segments:
            parent_segment.time_vector = None
            self.add_recording_segment(
                DecimateRecordingSegment(
                    parent_segment,
                    resample_rate,
                    self._orig_samp_freq,
                    decimation_frame_step,
                    decimation_frame_start,
                    dtype,
                )
            )

        self._kwargs = dict(
            recording=recording,
            decimation_frame_step=decimation_frame_step,
            decimation_frame_start=decimation_frame_start,
        )


class DecimateRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        parent_recording_segment,
        resample_rate,
        parent_rate,
        decimation_frame_step,
        decimation_frame_start,
        dtype,
    ):
        if parent_recording_segment.t_start is None:
            new_t_start = None
        else:
            new_t_start = parent_recording_segment.t_start + decimation_frame_start / parent_rate

        # Do not use BasePreprocessorSegment bcause we have to reset the sampling rate!
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency=resample_rate,
            t_start=new_t_start,
        )
        self._parent_segment = parent_recording_segment
        self._decimation_frame_step = decimation_frame_step
        self._decimation_frame_start = decimation_frame_start
        self._dtype = dtype

    def get_num_samples(self):
        return len(
            range(self._decimation_frame_start, self._parent_segment.get_num_samples(), self._decimation_frame_step)
        )

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        # Account for offset and end when querying parent traces
        parent_start_frame = self._decimation_frame_start + start_frame * self._decimation_frame_step
        parent_end_frame = parent_start_frame + (end_frame - start_frame) * self._decimation_frame_step

        # And now we can decimate without offsetting
        return self._parent_segment.get_traces(
            parent_start_frame,
            parent_end_frame,
            channel_indices,
        )[
            :: self._decimation_frame_step
        ].astype(self._dtype)


decimate = define_function_from_class(source_class=DecimateRecording, name="decimate")
