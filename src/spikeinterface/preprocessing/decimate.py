import warnings

import numpy as np
from spikeinterface.core.core_tools import (
    define_function_handling_dict_from_class,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from ._resampling_tools import get_polyphase_filter
from spikeinterface.core import BaseRecordingSegment, get_chunk_with_margin


class DecimateRecording(BasePreprocessor):
    """
    Decimate the recording extractor traces.

    By default this uses simple array slicing
    (``<parent_traces>[<decimation_offset>::<decimation_factor>]``), which is fast but applies no
    anti-aliasing filter and so might introduce aliasing, or skip across signal of interest. Set
    `antialias=True` to low-pass filter before downsampling using ``scipy.signal.resample_poly`` (the
    same anti-aliased decimation used by ``spikeinterface.preprocessing.ResampleRecording``).

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
    antialias : bool | None, default: None
        If True, apply an anti-aliasing low-pass filter before downsampling, using
        ``scipy.signal.resample_poly`` with a Kaiser-windowed FIR filter. If False,
        traces are downsampled by plain array slicing with no filtering, and `margin_ms`
        is ignored. If omitted or None, currently behaves as False and emits a FutureWarning:
        a future release will enable antialiasing by default. Pass True or False explicitly
        to select the behavior and silence the transition warning.
    margin_ms : float | None, default: None
        Additional context in ms on each side of a chunk. Only used when `antialias=True`.
        If None, use the FIR filter's finite support. An explicit nonnegative value requests
        at least that much context; filter support and sample-grid alignment are always retained.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used.

    Returns
    -------
    decimate_recording: DecimateRecording
        The decimated recording extractor object. With `antialias=False` the full traces of the
        child recording segment correspond to the traces of the parent segment as follows:
            ```<decimated_traces> = <parent_traces>[<decimation_offset>::<decimation_factor>]```

    """

    def __init__(
        self,
        recording,
        decimation_factor,
        decimation_offset=0,
        antialias=None,
        margin_ms=None,
        dtype=None,
    ):
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        if not isinstance(decimation_factor, int) or decimation_factor <= 0:
            raise ValueError(f"Expecting strictly positive integer for `decimation_factor` arg")
        self._decimation_factor = decimation_factor
        if not isinstance(decimation_offset, int) or decimation_offset < 0:
            raise ValueError("Expecting a nonnegative integer for `decimation_offset` arg")
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

        # fix_dtype doesn't always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str

        if antialias is None:
            warnings.warn(
                "The default for `antialias` will change to True in a future release. "
                "Currently, decimation uses slicing without an anti-aliasing filter. "
                "Pass antialias=True to enable the filter, "
                "or antialias=False to explicitly retain slicing.",
                FutureWarning,
                stacklevel=2,
            )
            antialias = False

        if antialias:
            filter_coefficients, margin = get_polyphase_filter(self._orig_samp_freq, 1, decimation_factor, margin_ms)
        else:
            filter_coefficients, margin = None, 0

        BasePreprocessor.__init__(self, recording, sampling_frequency=decimated_sampling_frequency, dtype=dtype)

        for parent_segment in recording.segments:
            self.add_recording_segment(
                DecimateRecordingSegment(
                    parent_segment,
                    decimated_sampling_frequency,
                    self._orig_samp_freq,
                    decimation_factor,
                    decimation_offset,
                    self._dtype,
                    antialias,
                    margin,
                    filter_coefficients,
                )
            )

        self._kwargs = dict(
            recording=recording,
            decimation_factor=decimation_factor,
            decimation_offset=decimation_offset,
            antialias=antialias,
            margin_ms=margin_ms,
            dtype=dtype,
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
        antialias=False,
        margin=0,
        filter_coefficients=None,
    ):
        if parent_recording_segment._time_vector is not None:
            time_vector = parent_recording_segment._time_vector[decimation_offset::decimation_factor]
            decimated_sampling_frequency = None
            t_start = None
        else:
            time_vector = None
            t_start = parent_recording_segment._t_start
            if decimation_offset:
                t_start = (0.0 if t_start is None else t_start) + decimation_offset / parent_rate

        # Do not use BasePreprocessorSegment bcause we have to reset the sampling rate!
        BaseRecordingSegment.__init__(
            self, sampling_frequency=decimated_sampling_frequency, t_start=t_start, time_vector=time_vector
        )
        self._parent_segment = parent_recording_segment
        self._decimation_factor = decimation_factor
        self._decimation_offset = decimation_offset
        self._dtype = dtype
        self._antialias = antialias
        self._margin = margin
        self._filter_coefficients = filter_coefficients

    def get_num_samples(self):
        parent_n_samp = self._parent_segment.get_num_samples()
        assert self._decimation_offset < parent_n_samp  # Sanity check (already enforced). Formula changes otherwise
        return (parent_n_samp - self._decimation_offset + self._decimation_factor - 1) // self._decimation_factor

    def get_traces(self, start_frame, end_frame, channel_indices):
        if not self._antialias:
            # Simple array slicing, no anti-aliasing filter.
            parent_start_frame = self._decimation_offset + start_frame * self._decimation_factor
            parent_end_frame = parent_start_frame + (end_frame - start_frame) * self._decimation_factor
            return self._parent_segment.get_traces(
                parent_start_frame,
                parent_end_frame,
                channel_indices,
            )[
                :: self._decimation_factor
            ].astype(self._dtype)

        return get_polyphase_resampled_traces(
            self._parent_segment,
            start_frame,
            end_frame,
            channel_indices,
            1,
            self._decimation_factor,
            self._margin,
            self._dtype,
            self._filter_coefficients,
            decimation_offset=self._decimation_offset,
        )


def get_polyphase_resampled_traces(
    parent_segment,
    start_frame,
    end_frame,
    channel_indices,
    up,
    down,
    margin,
    dtype,
    filter_coefficients,
    decimation_offset=0,
):
    """Resample a chunk, with reflected boundary padding."""
    from scipy.signal import resample_poly

    if end_frame <= start_frame:
        return parent_segment.get_traces(0, 0, channel_indices).astype(dtype)

    parent_start_frame = decimation_offset + (start_frame // up) * down
    parent_end_frame = decimation_offset + ((end_frame + up - 1) // up) * down
    parent_traces, left_margin, _ = get_chunk_with_margin(
        parent_segment,
        parent_start_frame,
        parent_end_frame,
        channel_indices,
        margin,
        add_reflect_padding=True,
    )
    working_dtype = np.result_type(parent_traces.dtype, dtype, np.float32)
    traces = resample_poly(
        parent_traces.astype(working_dtype, copy=False),
        up,
        down,
        axis=0,
        window=filter_coefficients.astype(working_dtype, copy=False),
    )
    start_drop = start_frame % up + left_margin * up // down
    traces = traces[start_drop : start_drop + end_frame - start_frame]
    return _cast_resampled_traces(traces, dtype)


def _cast_resampled_traces(traces, dtype):
    """Reject nonfinite output and round and saturate integer conversions."""
    if not np.all(np.isfinite(traces)):
        raise ValueError("Resampling produced nonfinite values. Check the input traces and resampling parameters.")

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        rounded = np.rint(traces)
        limits = np.iinfo(dtype)
        below = rounded <= limits.min
        above = rounded >= limits.max

        # Assign saturated endpoints after casting because apparently float64 can't
        # represent int64.max exactly.
        rounded[below | above] = 0
        result = rounded.astype(dtype)
        result[below] = limits.min
        result[above] = limits.max
        return result

    if np.issubdtype(dtype, np.floating) and np.any(np.abs(traces) > np.finfo(dtype).max):
        raise ValueError(f"Resampled values exceed the finite range of {dtype}.")
    return traces.astype(dtype, copy=False)


decimate = define_function_handling_dict_from_class(source_class=DecimateRecording, name="decimate")
