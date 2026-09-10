import numpy as np
from spikeinterface.core.core_tools import (
    define_function_handling_dict_from_class,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from ._decimation_tools import get_balanced_decimation_factors
from spikeinterface.core import BaseRecordingSegment, get_chunk_with_margin


class DecimateRecording(BasePreprocessor):
    """
    Decimate the recording extractor traces.

    By default this uses simple array slicing
    (``<parent_traces>[<decimation_offset>::<decimation_factor>]``), which is fast but applies no
    anti-aliasing filter and so might introduce aliasing, or skip across signal of interest. Set
    `antialias=True` to low-pass filter before downsampling using ``scipy.signal.decimate`` (the
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
    antialias : bool, default: False
        If True, apply an anti-aliasing low-pass filter before downsampling, using
        ``scipy.signal.decimate``. When `decimation_factor` exceeds 13, the decimation is
        automatically performed in several balanced sub-13 passes (e.g. a factor of 48 is applied
        as 8 then 6), as scipy recommends, to keep the IIR anti-aliasing filter stable. If False
        (the default), traces are downsampled by plain array slicing with no filtering, and
        `margin_ms` is ignored.
    margin_ms : float, default: 100.0
        Margin in ms used on each side of every chunk to limit edge effects of the anti-aliasing
        filter. Only used when `antialias=True`. The margin is internally rounded up to a whole
        number of output samples so the filtered, downsampled traces stay aligned across chunks.
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
        antialias=False,
        margin_ms=100.0,
        dtype=None,
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

        # fix_dtype doesn't always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str

        decimation_factors = get_balanced_decimation_factors(decimation_factor) if antialias else None

        # Margin (in parent samples) to limit anti-aliasing filter edge effects.
        margin = int(margin_ms * self._orig_samp_freq / 1000)

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
                    decimation_factors,
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
        decimation_factors=None,
    ):
        if parent_recording_segment._time_vector is not None:
            time_vector = parent_recording_segment._time_vector[decimation_offset::decimation_factor]
            decimated_sampling_frequency = None
            t_start = None
        else:
            time_vector = None
            if parent_recording_segment._t_start is None:
                t_start = None
            else:
                t_start = parent_recording_segment._t_start + (decimation_offset / parent_rate)

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
        self._decimation_factors = decimation_factors if decimation_factors is not None else [decimation_factor]

    def get_num_samples(self):
        parent_n_samp = self._parent_segment.get_num_samples()
        assert self._decimation_offset < parent_n_samp  # Sanity check (already enforced). Formula changes otherwise
        return int(np.ceil((parent_n_samp - self._decimation_offset) / self._decimation_factor))

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

        # Anti-aliased decimation as a cascade of balanced scipy.signal.decimate passes.
        return get_antialiased_decimated_traces(
            self._parent_segment,
            start_frame,
            end_frame,
            channel_indices,
            self._decimation_factor,
            self._decimation_factors,
            self._margin,
            self._dtype,
            decimation_offset=self._decimation_offset,
        )


def get_antialiased_decimated_traces(
    parent_segment,
    start_frame,
    end_frame,
    channel_indices,
    decimation_factor,
    decimation_factors,
    margin,
    dtype,
    decimation_offset=0,
):
    """
    Fetch a margined chunk from `parent_segment` and decimate it by `decimation_factor`, applied
    as a cascade of the balanced `decimation_factors` passes of ``scipy.signal.decimate``.

    The margin is rounded up to a multiple of the total `decimation_factor` so that
    ``left_margin // decimation_factor`` is exact; combined with scipy's default
    ``zero_phase=True`` (output sample i maps to filtered input sample i * factor), this keeps the
    downsampled traces aligned across chunks (a chunked read matches a full read). Exactly
    ``end_frame - start_frame`` decimated samples are returned.

    Parameters
    ----------
    parent_segment : BaseRecordingSegment
        The parent segment to read (full-rate) traces from.
    start_frame, end_frame : int
        Output (decimated) frame range to return.
    channel_indices : slice | list | np.ndarray | None
        Channels to read, forwarded to the parent segment.
    decimation_factor : int
        The total decimation factor (the product of `decimation_factors`).
    decimation_factors : list[int]
        The per-pass sub-factors (each <= 13), e.g. from `get_balanced_decimation_factors`.
    margin : int
        Margin in parent samples used to limit anti-aliasing filter edge effects. Rounded up
        internally to a multiple of `decimation_factor`.
    dtype : np.dtype | str
        Output dtype. Integer output is rounded and clipped to its range.
    decimation_offset : int, default: 0
        Index of the first parent frame, applied to the first output sample only.
    """
    from scipy import signal

    q = decimation_factor
    parent_start_frame = decimation_offset + start_frame * q
    parent_end_frame = parent_start_frame + (end_frame - start_frame) * q
    # Round the margin up to a multiple of q so that left_margin // q is exact.
    margin = ((margin + q - 1) // q) * q
    parent_traces, left_margin, right_margin = get_chunk_with_margin(
        parent_segment,
        parent_start_frame,
        parent_end_frame,
        channel_indices,
        margin,
        add_reflect_padding=True,
    )
    working_dtype = np.result_type(parent_traces.dtype, dtype, np.float32)
    decimated_traces = parent_traces.astype(working_dtype, copy=False)
    for sub_q in decimation_factors:
        decimated_traces = signal.decimate(decimated_traces, q=sub_q, axis=0)
    start_drop = left_margin // q
    n_out = end_frame - start_frame
    decimated_traces = decimated_traces[start_drop : start_drop + n_out]
    return _cast_resampled_traces(decimated_traces, dtype)


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
