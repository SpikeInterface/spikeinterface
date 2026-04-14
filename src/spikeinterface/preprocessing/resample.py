import numpy as np
import warnings

from spikeinterface.core.core_tools import (
    define_function_handling_dict_from_class,
    recursive_key_finder,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from spikeinterface.core import get_chunk_with_margin, BaseRecordingSegment


class ResampleRecording(BasePreprocessor):
    """
    Resample the recording extractor traces.

    If the original sampling rate is multiple of the resample_rate, it will use
    the signal.decimate method from scipy. In other cases, it uses signal.resample. In the
    later case, the resulting signal can have issues on the edges, mainly on the
    rightmost.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    resample_rate : int
        The resampling frequency
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
    margin_ms : float, default: 100.0
        Margin in ms for computations, will be used to decrease edge effects.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used.
    skip_checks : bool, default: False
        If True, checks on sampling frequencies and cutoff filter frequencies are skipped

    Returns
    -------
    resample_recording : ResampleRecording
        The resampled recording extractor object.

    """

    def __init__(
        self,
        recording,
        resample_rate,
        gap_tolerance_ms=None,
        margin_ms=100.0,
        dtype=None,
        skip_checks=False,
    ):
        # Floating point resampling rates can lead to unexpected results, avoid actively
        msg = "Non integer resampling rates can lead to unexpected results."
        assert isinstance(resample_rate, (int, np.integer)), msg
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        self._resample_rate = resample_rate
        self._sampling_frequency = resample_rate
        # fix_dtype not always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str
        # Ensure that the requested resample rate is doable:
        if skip_checks:
            assert check_nyquist(recording, resample_rate), "The requested resample rate would induce errors!"

        # Get a margin to avoid issues later
        margin = int(margin_ms * recording.get_sampling_frequency() / 1000)

        BasePreprocessor.__init__(self, recording, sampling_frequency=resample_rate, dtype=dtype)
        for parent_segment in recording.segments:
            self.add_recording_segment(
                ResampleRecordingSegment(
                    parent_segment,
                    resample_rate,
                    recording.get_sampling_frequency(),
                    margin,
                    dtype,
                    gap_tolerance_ms,
                )
            )

        self._kwargs = dict(
            recording=recording,
            resample_rate=resample_rate,
            gap_tolerance_ms=gap_tolerance_ms,
            margin_ms=margin_ms,
            dtype=dtype,
            skip_checks=skip_checks,
        )


class ResampleRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        parent_recording_segment,
        resample_rate,
        parent_rate,
        margin,
        dtype,
        gap_tolerance_ms=None,
    ):
        self._resample_rate = resample_rate
        self._parent_segment = parent_recording_segment
        self._parent_rate = parent_rate
        self._margin = margin
        self._dtype = dtype
        self._has_gaps = False

        # Compute time_vector or t_start, following the pattern from DecimateRecordingSegment.
        # Do not use BasePreprocessorSegment because we have to reset the sampling rate!
        if parent_recording_segment.time_vector is not None:
            parent_tv = np.asarray(parent_recording_segment.time_vector)

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
                    int((sec_boundaries_parent[k, 1] - sec_boundaries_parent[k, 0]) / parent_rate * resample_rate)
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

            # Compute time_vector
            n_out = int(len(parent_tv) / parent_rate * resample_rate)

            if parent_rate % resample_rate == 0:
                q_int = int(parent_rate / resample_rate)
                if not self._has_gaps:
                    time_vector = parent_tv[::q_int][:n_out]
                else:
                    # Section-wise slicing to keep time_vector consistent
                    # with _sec_boundaries_output
                    tv_pieces = []
                    for k in range(K):
                        p_start, p_end = sec_boundaries_parent[k]
                        n_out_k = sec_n_out[k]
                        if n_out_k == 0:
                            continue
                        tv_pieces.append(parent_tv[p_start:p_end:q_int][:n_out_k])
                    time_vector = np.concatenate(tv_pieces)
            elif not self._has_gaps:
                # Non-integer ratio, no gaps: existing fast path
                warnings.warn(
                    "Resampling with a non-integer ratio requires interpolating the time_vector. "
                    "An integer ratio (parent_rate / resample_rate) is more performant."
                )
                parent_indices = np.linspace(0, len(parent_tv) - 1, n_out)
                time_vector = np.interp(parent_indices, np.arange(len(parent_tv)), parent_tv)
            else:
                # Non-integer ratio with gaps: per-section interpolation
                warnings.warn(
                    "Resampling with a non-integer ratio requires interpolating the time_vector. "
                    "An integer ratio (parent_rate / resample_rate) is more performant."
                )
                tv_pieces = []
                for k in range(K):
                    p_start, p_end = sec_boundaries_parent[k]
                    n_out_k = sec_n_out[k]
                    if n_out_k == 0:
                        continue
                    sec_parent_tv = parent_tv[p_start:p_end]
                    sec_len = p_end - p_start
                    sec_indices = np.linspace(0, sec_len - 1, n_out_k)
                    tv_pieces.append(np.interp(sec_indices, np.arange(sec_len), sec_parent_tv))
                time_vector = np.concatenate(tv_pieces)

            BaseRecordingSegment.__init__(self, sampling_frequency=None, t_start=None, time_vector=time_vector)
        else:
            BaseRecordingSegment.__init__(
                self, sampling_frequency=resample_rate, t_start=parent_recording_segment.t_start
            )

    def get_num_samples(self):
        if self.time_vector is not None:
            return len(self.time_vector)
        return int(self._parent_segment.get_num_samples() / self._parent_rate * self._resample_rate)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if self._has_gaps:
            return self._get_traces_gapped(start_frame, end_frame, channel_indices)

        # Original code path: no gaps (or no time_vector)
        # get parent traces with margin
        parent_start_frame, parent_end_frame = [
            int((frame / self._resample_rate) * self._parent_rate) for frame in [start_frame, end_frame]
        ]
        parent_traces, left_margin, right_margin = get_chunk_with_margin(
            self._parent_segment,
            parent_start_frame,
            parent_end_frame,
            channel_indices,
            self._margin,
            add_reflect_padding=True,
            dtype=np.float32,
        )
        # get left and right margins for the resampled case
        left_margin_rs, right_margin_rs = [
            int((margin / self._parent_rate) * self._resample_rate) for margin in [left_margin, right_margin]
        ]

        # get the size for the resampled traces in case of resample:
        num = int((end_frame + right_margin_rs) - (start_frame - left_margin_rs))

        # Decimate can misbehave on some cases, while resample always looks nice enough.
        # Check which method to use:
        from scipy import signal

        if np.mod(self._parent_rate, self._resample_rate) == 0:
            # Ratio between sampling frequencies
            q = int(self._parent_rate / self._resample_rate)
            # Decimate can have issues for some cases, returning NaNs
            resampled_traces = signal.decimate(parent_traces, q=q, axis=0)
            # If that's the case, use signal.resample
            if np.any(np.isnan(resampled_traces)):
                resampled_traces = signal.resample(parent_traces, num, axis=0)
        else:
            resampled_traces = signal.resample(parent_traces, num, axis=0)

        # now take care of the edges
        resampled_traces = resampled_traces[left_margin_rs : num - right_margin_rs]
        return resampled_traces.astype(self._dtype)

    def _get_traces_gapped(self, start_frame, end_frame, channel_indices):
        """Resample traces section-by-section, avoiding FFT processing across gaps."""
        from scipy import signal

        # Determine the post-indexing channel count via a 1-sample parent fetch.
        # channel_indices may be a slice, list, ndarray, or None, so we cannot
        # simply use len(channel_indices).
        n_channels = self._parent_segment.get_traces(0, 1, channel_indices).shape[1]

        # Pre-allocate the output buffer.
        result = np.empty((end_frame - start_frame, n_channels), dtype=self._dtype)

        if start_frame == end_frame:
            return result

        # Find which sections overlap [start_frame, end_frame) in output space.
        # _sec_boundaries_output[k] = [out_start_k, out_end_k)
        sec_ends = self._sec_boundaries_output[:, 1]
        sec_starts = self._sec_boundaries_output[:, 0]
        first_sec = int(np.searchsorted(sec_ends, start_frame, side="right"))
        last_sec = int(np.searchsorted(sec_starts, end_frame, side="left")) - 1
        first_sec = max(first_sec, 0)
        last_sec = min(last_sec, len(self._sec_n_out) - 1)

        is_integer_ratio = (self._parent_rate % self._resample_rate) == 0

        pos = 0
        for k in range(first_sec, last_sec + 1):
            out_start_k = int(self._sec_boundaries_output[k, 0])
            out_end_k = int(self._sec_boundaries_output[k, 1])
            par_start_k = int(self._sec_boundaries_parent[k, 0])
            par_end_k = int(self._sec_boundaries_parent[k, 1])
            sec_n_parent = par_end_k - par_start_k
            sec_n_output = int(self._sec_n_out[k])

            if sec_n_output == 0:
                continue

            # Clip the output range to the requested [start_frame, end_frame)
            local_out_start = max(start_frame, out_start_k) - out_start_k
            local_out_end = min(end_frame, out_end_k) - out_start_k

            if local_out_end <= local_out_start:
                continue

            # Map within-section output frames to within-section parent frames
            local_par_start = int((local_out_start / self._resample_rate) * self._parent_rate)
            local_par_end = int((local_out_end / self._resample_rate) * self._parent_rate)
            local_par_start = max(0, min(local_par_start, sec_n_parent))
            local_par_end = max(0, min(local_par_end, sec_n_parent))

            # Apply margin within section boundaries only (do not cross gaps)
            left_margin = min(self._margin, local_par_start)
            right_margin = min(self._margin, sec_n_parent - local_par_end)

            par_fetch_start = par_start_k + local_par_start - left_margin
            par_fetch_end = par_start_k + local_par_end + right_margin

            # Fetch parent traces for this section's sub-chunk
            parent_traces = self._parent_segment.get_traces(par_fetch_start, par_fetch_end, channel_indices).astype(
                np.float32
            )

            # Apply reflect padding if margin was truncated at section edge
            pad_left = self._margin - left_margin
            pad_right = self._margin - right_margin
            if pad_left > 0 or pad_right > 0:
                parent_traces = np.pad(parent_traces, [(pad_left, pad_right), (0, 0)], mode="reflect")
                left_margin = self._margin
                right_margin = self._margin

            # Compute resampled margins
            left_margin_rs = int((left_margin / self._parent_rate) * self._resample_rate)
            right_margin_rs = int((right_margin / self._parent_rate) * self._resample_rate)

            # Total output samples including margins
            chunk_len = int(local_out_end - local_out_start)
            num = chunk_len + left_margin_rs + right_margin_rs

            # Resample this section
            if is_integer_ratio:
                q = int(self._parent_rate / self._resample_rate)
                resampled = signal.decimate(parent_traces, q=q, axis=0)
                if np.any(np.isnan(resampled)):
                    resampled = signal.resample(parent_traces, num, axis=0)
            else:
                resampled = signal.resample(parent_traces, num, axis=0)

            # Trim margins and write directly into the pre-allocated buffer.
            # Clamp to the remaining space in case decimate's output length
            # differs from `num` by a rounding sample.
            trimmed = resampled[left_margin_rs : num - right_margin_rs]
            write_len = min(len(trimmed), result.shape[0] - pos)
            result[pos : pos + write_len] = trimmed[:write_len]
            pos += write_len

        # Return only the filled portion (normally equals end_frame - start_frame).
        return result[:pos]


resample = define_function_handling_dict_from_class(source_class=ResampleRecording, name="resample")


# Some helpers to do checks
def check_nyquist(recording, resample_rate):
    # Check that the original and requested sampling rates will not induce aliasing
    # Basic test, compare the sampling frequency with the resample rate
    sampling_frequency_check = recording.get_sampling_frequency() / 2 > resample_rate
    # Check that the signal, if it has been filtered, is still not violating
    if recording.is_filtered():
        # Check if we have access to the highcut frequency
        freq_max = list(recursive_key_finder(recording, "freq_max"))
        if freq_max:
            # Given that there might be more than one filter applied, keep the lowest
            freq_max = min(freq_max)
            lowpass_cutoff_check = freq_max / 2 > resample_rate
        else:
            # If has been filterd but unknown high cutoff, give warning and asume the best
            warnings.warn("The recording is filtered, but we can't ensure that it complies with the Nyquist limit.")
            lowpass_cutoff_check = True
    else:
        # If it hasn't been filtered, we only depend on the previous test
        warnings.warn(
            "The recording is not filtered, so cutoff frequencies cannot be checked. " "Use resampling with caution"
        )
        lowpass_cutoff_check = True
    return all([sampling_frequency_check, lowpass_cutoff_check])
