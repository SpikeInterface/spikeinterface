import numpy as np
import warnings

from spikeinterface.core.core_tools import define_function_handling_dict_from_class

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from ._resampling_tools import get_resampling_factors, get_polyphase_filter
from .decimate import get_polyphase_resampled_traces
from spikeinterface.core import BaseRecordingSegment
from spikeinterface.core.frameslicerecording import FrameSliceRecordingSegment


class ResampleRecording(BasePreprocessor):
    """
    Resample the recording extractor traces.

    Uses ``scipy.signal.resample_poly`` with a Kaiser-windowed FIR filter for both
    downsampling and upsampling. Detected gaps are handled section by section.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    resample_rate : int | float
        The requested sampling frequency. The closest output/input rate ratio with
        denominator at most `max_denominator` is selected. The output reports the
        achieved rate, while the requested rate is retained in serialization kwargs.
        A relative difference exceeding 1e-12 emits a warning.
    gap_tolerance_ms : float | None, default: None
        Maximum acceptable gap size in milliseconds for automatic segmentation.

        **Default behavior (None)**: If timestamp gaps are detected in the parent
        recording's time vector, an error is raised with a detailed gap report.
        This ensures users are aware of data discontinuities rather than silently
        producing incorrect results.

        **Opt-in segmentation**: Provide a value to automatically handle gaps via
        section-wise resampling. Gaps larger than this threshold trigger section
        splitting; gaps smaller than the threshold are ignored (data treated as
        continuous). Within each contiguous section, resampling proceeds correctly.

        In all cases, deviations smaller than 1.5 sample periods are never treated
        as gaps, since sub-sample jitter and floating-point noise in time vectors
        cannot represent dropped samples.

        Examples:
        - None (default): Error on any detected gaps
        - 0.0: Strict mode — split on any gap >= 1.5 sample periods
        - 1.0: Tolerate gaps up to 1 ms, split on larger gaps
        - 100.0: Only major pauses (>100 ms) create sections
    margin_ms : float | None, default: None
        Additional context in ms on each side of a chunk. If None, use the FIR filter's
        finite support. An explicit nonnegative value requests at least that much context;
        filter support is always retained within each section.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used.
        Integer output is rounded and clipped to the dtype range. Nonfinite resampled
        output raises a ValueError before conversion.

    max_denominator : int, default: 10000
        Maximum denominator of the rational output/input rate ratio. Increasing this can
        improve rate accuracy, but can also increase filter length and the aligned input
        span needed for a chunk.

    Returns
    -------
    resample_recording : ResampleRecording
        The resampled recording extractor object.

    Notes
    -----
    Each section returns ``ceil(num_input_samples * up / down)`` samples, matching SciPy
    and ``decimate()``. Output timestamps use the same rational grid as the traces.
    Explicit parent timestamps are sampled or interpolated within each section.
    Output positions beyond the last input sample extrapolate its timestamp by less
    than one nominal input period.

    For example, resampling from 30000.01 Hz to a requested 2500 Hz with the default
    denominator limit selects up=1 and down=12. The output reports 2500.000833333333 Hz,
    and a warning reports the difference of approximately +0.333333 ppm.

    """

    def __init__(
        self,
        recording,
        resample_rate,
        gap_tolerance_ms=None,
        margin_ms=None,
        dtype=None,
        max_denominator=10000,
    ):
        self._orig_samp_freq = recording.get_sampling_frequency()
        up, down, achieved_rate = get_resampling_factors(self._orig_samp_freq, resample_rate, max_denominator)
        self._resample_rate = achieved_rate
        # fix_dtype not always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str

        filter_coefficients, margin = get_polyphase_filter(self._orig_samp_freq, up, down, margin_ms)

        BasePreprocessor.__init__(self, recording, sampling_frequency=achieved_rate, dtype=dtype)
        for parent_segment in recording.segments:
            self.add_recording_segment(
                ResampleRecordingSegment(
                    parent_segment,
                    achieved_rate,
                    recording.get_sampling_frequency(),
                    margin,
                    dtype,
                    gap_tolerance_ms,
                    up,
                    down,
                    filter_coefficients,
                )
            )

        self._kwargs = dict(
            recording=recording,
            resample_rate=resample_rate,
            gap_tolerance_ms=gap_tolerance_ms,
            margin_ms=margin_ms,
            dtype=dtype,
            max_denominator=max_denominator,
        )


class ResampleRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        parent_recording_segment,
        resample_rate,
        parent_rate,
        margin,
        dtype,
        gap_tolerance_ms,
        up,
        down,
        filter_coefficients,
    ):
        self._resample_rate = resample_rate
        self._parent_segment = parent_recording_segment
        self._parent_rate = parent_rate
        self._margin = margin
        self._dtype = dtype
        self._has_gaps = False
        self._up = up
        self._down = down
        self._filter_coefficients = filter_coefficients

        # Compute time_vector or t_start, following the pattern from DecimateRecordingSegment.
        # Do not use BasePreprocessorSegment because we have to reset the sampling rate!
        if parent_recording_segment._time_vector is not None:
            parent_tv = np.asarray(parent_recording_segment._time_vector)

            # Detect gaps in the parent time vector.
            # A true gap means at least one dropped sample, so dt >= 2 * expected_dt.
            # Use 1.5 * expected_dt as the minimum threshold to avoid false positives
            # from floating-point jitter while catching any real dropped samples.
            expected_dt = 1.0 / parent_rate
            min_gap_threshold = 1.5 * expected_dt
            if gap_tolerance_ms is not None:
                detection_threshold = max(min_gap_threshold, gap_tolerance_ms / 1000.0)
            else:
                detection_threshold = min_gap_threshold

            diffs = np.diff(parent_tv)
            gap_indices = np.flatnonzero(diffs > detection_threshold)

            if len(gap_indices) > 0 and gap_tolerance_ms is None:
                gap_sizes_ms = diffs[gap_indices] * 1000
                gap_positions_s = parent_tv[gap_indices]
                raise ValueError(
                    f"Detected {len(gap_indices)} timestamp gap(s) in the parent "
                    f"recording's time vector.\n"
                    f"  Gap sizes (ms): {gap_sizes_ms}\n"
                    f"  Gap positions (seconds): {gap_positions_s}\n"
                    f"  Gap positions (parent sample indices): {gap_indices}\n"
                    f"To handle gaps automatically via section-wise resampling, "
                    f"pass gap_tolerance_ms=<threshold>. Gaps larger than the "
                    f"threshold will trigger section splitting; smaller gaps are "
                    f"treated as continuous."
                )

            # Build section boundaries: contiguous runs of samples between gaps.
            # We call these "sections" (not "segments") to avoid confusion with
            # the Segment concept in SpikeInterface/neo.
            if len(gap_indices) == 0:
                sec_boundaries_parent = np.array([[0, len(parent_tv)]], dtype=np.int64)
            else:
                self._has_gaps = True
                starts = np.concatenate([[0], gap_indices + 1]).astype(np.int64)
                ends = np.concatenate([gap_indices + 1, [len(parent_tv)]]).astype(np.int64)
                sec_boundaries_parent = np.column_stack([starts, ends])

            # Compute per-section output sample counts and cumulative boundaries.
            K = len(sec_boundaries_parent)
            sec_n_out = np.array(
                [
                    (int(sec_boundaries_parent[k, 1] - sec_boundaries_parent[k, 0]) * up + down - 1) // down
                    for k in range(K)
                ],
                dtype=np.int64,
            )
            sec_cumstart = np.zeros(K, dtype=np.int64)
            sec_cumstart[1:] = np.cumsum(sec_n_out[:-1])
            sec_boundaries_output = np.column_stack([sec_cumstart, sec_cumstart + sec_n_out])

            self._sec_boundaries_parent = sec_boundaries_parent
            self._sec_boundaries_output = sec_boundaries_output
            self._sec_n_out = sec_n_out

            tv_pieces = [
                _resample_time_vector(parent_tv[p_start:p_end], up, down, parent_rate)
                for p_start, p_end in sec_boundaries_parent
            ]
            time_vector = np.concatenate(tv_pieces) if self._has_gaps else tv_pieces[0]

            BaseRecordingSegment.__init__(self, sampling_frequency=None, t_start=None, time_vector=time_vector)
        else:
            BaseRecordingSegment.__init__(
                self, sampling_frequency=resample_rate, t_start=parent_recording_segment._t_start
            )

    def get_num_samples(self):
        if self._time_vector is not None:
            return len(self._time_vector)
        n = self._parent_segment.get_num_samples()
        return (n * self._up + self._down - 1) // self._down

    def get_traces(self, start_frame, end_frame, channel_indices):
        if end_frame <= start_frame:
            return self._parent_segment.get_traces(0, 0, channel_indices).astype(self._dtype)
        if self._has_gaps:
            return self._get_traces_gapped(start_frame, end_frame, channel_indices)

        return self._get_resampled_traces(self._parent_segment, start_frame, end_frame, channel_indices)

    def _get_resampled_traces(self, parent_segment, start_frame, end_frame, channel_indices):
        return get_polyphase_resampled_traces(
            parent_segment,
            start_frame,
            end_frame,
            channel_indices,
            self._up,
            self._down,
            self._margin,
            self._dtype,
            self._filter_coefficients,
        )

    def _get_traces_gapped(self, start_frame, end_frame, channel_indices):
        """Resample each section with margins bounded by its own samples."""
        n_channels = self._parent_segment.get_traces(0, 1, channel_indices).shape[1]
        result = np.empty((end_frame - start_frame, n_channels), dtype=self._dtype)
        if start_frame == end_frame:
            return result

        sec_starts = self._sec_boundaries_output[:, 0]
        sec_ends = self._sec_boundaries_output[:, 1]
        first_sec = int(np.searchsorted(sec_ends, start_frame, side="right"))
        stop_sec = int(np.searchsorted(sec_starts, end_frame, side="left"))

        for k in range(first_sec, stop_sec):
            out_start = max(start_frame, int(sec_starts[k]))
            out_end = min(end_frame, int(sec_ends[k]))
            if out_start >= out_end:
                continue

            par_start, par_end = self._sec_boundaries_parent[k]
            section = FrameSliceRecordingSegment(self._parent_segment, int(par_start), int(par_end))
            result[out_start - start_frame : out_end - start_frame] = self._get_resampled_traces(
                section, out_start - int(sec_starts[k]), out_end - int(sec_starts[k]), channel_indices
            )

        return result


def _resample_time_vector(parent_times, up, down, parent_rate):
    """Map the rational sample grid onto timestamps without crossing section boundaries."""
    if up == 1:
        return parent_times[::down]
    n_out = (len(parent_times) * up + down - 1) // down
    positions = np.arange(n_out, dtype=np.int64) * down
    left = positions // up
    weight = (positions % up) / up
    right = np.minimum(left + 1, len(parent_times) - 1)
    intervals = parent_times[right] - parent_times[left]
    intervals[left == len(parent_times) - 1] = 1.0 / parent_rate
    return parent_times[left] + weight * intervals


resample = define_function_handling_dict_from_class(source_class=ResampleRecording, name="resample")
