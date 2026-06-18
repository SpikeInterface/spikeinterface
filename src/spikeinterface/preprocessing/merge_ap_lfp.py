import math
from typing import Callable, ClassVar, List, Union
import numpy as np

from ..core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin


class MergeApLfpRecording(BaseRecording):
    """
    Add cool description here.

    Parameters
    ----------
    ap_recording: BaseRecording
        The recording of the AP channels.
    lfp_recording: BaseRecording
        The recording of the LFP channels.
    ap_filter: Callable
        Transfer function of the filter used in the ap_recording.
        Takes the frequencies as parameter, and outputs the transfer function.
    lfp_filter: Callable
        Transfer function of the filter used in the lfp_recording.
        Takes the frequencies as parameter, and outputs the transfer function.
    margin: int
        The margin (in samples) to use when extracting the trace.

    Returns
    --------
    merged_ap_lfp_recording: MergeApLfpRecording
        The result of the merge of both channels (with the whole frequency spectrum).
    """

    def __init__(
        self,
        ap_recording: BaseRecording,
        lfp_recording: BaseRecording,
        ap_filter: Callable[[np.ndarray], np.ndarray],
        lfp_filter: Callable[[np.ndarray], np.ndarray],
        margin: int = 6_000,
    ) -> None:
        BaseRecording.__init__(self, ap_recording.sampling_frequency, ap_recording.channel_ids, ap_recording.dtype)
        ap_recording.copy_metadata(self)

        if ap_recording.has_scaleable_traces():
            ap_gain = ap_recording.get_property("gain_to_uV")
        else:
            ap_gain = np.ones(ap_recording.get_num_channels(), dtype=np.float32)
        if lfp_recording.has_scaleable_traces():
            lfp_gain = lfp_recording.get_property("gain_to_uV")
        else:
            lfp_gain = np.ones(lfp_recording.get_num_channels(), dtype=np.float32)

        for segment_index in range(ap_recording.get_num_segments()):
            ap_recording_segment = ap_recording._recording_segments[segment_index]
            lfp_recording_segment = lfp_recording._recording_segments[segment_index]
            self.add_recording_segment(
                MergeApLfpRecordingSegment(
                    ap_recording_segment,
                    lfp_recording_segment,
                    ap_filter,
                    lfp_filter,
                    margin,
                    lfp_gain / ap_gain,
                    ap_recording.get_dtype(),
                )
            )

        self._kwargs = {  # TODO: Is callable serializable? (missing ap_filter & lfp_filter at the moment)
            "ap_recording": ap_recording.to_dict(),
            "lfp_recording": lfp_recording.to_dict(),
            "margin": margin,
        }


class MergeApLfpRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        ap_recording_segment: BaseRecordingSegment,
        lfp_recording_segment: BaseRecordingSegment,
        ap_filter: Callable[[np.ndarray], np.ndarray],
        lfp_filter: Callable[[np.ndarray], np.ndarray],
        margin: int,
        lfp_to_ap_gain: np.ndarray,
        dtype,
    ) -> None:
        BaseRecordingSegment.__init__(self, ap_recording_segment.sampling_frequency, ap_recording_segment.t_start)

        self.ap_recording = ap_recording_segment
        self.lfp_recording = lfp_recording_segment
        self.ap_filter = ap_filter
        self.lfp_filter = lfp_filter
        self.margin = margin
        self.lfp_to_ap_gain = lfp_to_ap_gain
        self.dtype = dtype

        self.AP_TO_LFP = int(round(ap_recording_segment.sampling_frequency / lfp_recording_segment.sampling_frequency))

    def get_num_samples(self) -> int:
        # Trunk the recording to have a number of samples that is a multiple of 'AP_TO_LFP'.
        return self.ap_recording.get_num_samples() - (self.ap_recording.get_num_samples() % self.AP_TO_LFP)

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        from scipy.optimize import minimize
        import time

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        ap_traces, left_margin, right_margin = get_chunk_with_margin(
            self.ap_recording, start_frame, end_frame, channel_indices, self.margin + self.AP_TO_LFP
        )
        t15 = time.perf_counter()

        left_leftover = (self.AP_TO_LFP - (start_frame - left_margin) % self.AP_TO_LFP) % self.AP_TO_LFP
        left_margin -= left_leftover
        right_leftover = (end_frame + right_margin) % self.AP_TO_LFP
        right_margin -= right_leftover

        if right_leftover > 0:
            ap_traces = ap_traces[:-right_leftover]
        ap_traces = ap_traces[left_leftover:]

        lfp_traces = (
            self.lfp_recording.get_traces(
                (start_frame - left_margin) // self.AP_TO_LFP,
                (end_frame + right_margin) // self.AP_TO_LFP,
                channel_indices,
            )
            * self.lfp_to_ap_gain[channel_indices]
        )

        ap_fourier = np.fft.rfft(ap_traces, axis=0)
        lfp_fourier = np.fft.rfft(lfp_traces, axis=0)
        ap_freq = np.fft.rfftfreq(ap_traces.shape[0], d=1 / self.ap_recording.sampling_frequency)
        lfp_freq = np.fft.rfftfreq(lfp_traces.shape[0], d=1 / self.lfp_recording.sampling_frequency)

        ap_filter = self.ap_filter(ap_freq)
        lfp_filter = self.lfp_filter(lfp_freq)
        ap_filter[0] = lfp_filter[0] = 1.0  # Don't reconstruct 0 Hz.

        ap_fourier /= ap_filter[:, None]
        lfp_fourier /= lfp_filter[:, None]

        # Compute time shift between AP and LFP (TODO: Compute once and store?)
        freq_slice = slice(np.searchsorted(ap_freq, 100), np.searchsorted(ap_freq, 600))

        t_axis = np.arange(-2000, 2000, 60) * 1e-6
        errors = [
            _time_shift_error(t, ap_fourier[freq_slice, :], lfp_fourier[freq_slice, :], ap_freq[freq_slice])
            for t in t_axis
        ]
        shift_estimate = t_axis[np.argmin(errors)]

        minimization = minimize(
            _time_shift_error,
            method="Powell",
            x0=[shift_estimate],
            args=(ap_fourier[freq_slice, :], lfp_fourier[freq_slice, :], ap_freq[freq_slice]),
            bounds=[(shift_estimate - 1e-4, shift_estimate + 1e-4)],
            tol=1e-6,
        )
        shift_estimate = minimization.x[0]
        lfp_fourier /= np.exp(-2j * math.pi * lfp_freq[:, None] * shift_estimate)

        # Compute aliasing of high frequencies on LFP channels
        lfp_nyquist = self.lfp_recording.sampling_frequency / 2
        nyquist_index = len(lfp_freq)
        fourier_aliased = ap_fourier * np.exp(-2j * math.pi * ap_freq[:, None] * shift_estimate)
        fourier_aliased[:nyquist_index] = 0.0
        fourier_aliased *= self.lfp_filter(ap_freq)[:, None]
        traces_aliased = np.fft.irfft(fourier_aliased, axis=0)[:: self.AP_TO_LFP]
        fourier_aliased = np.fft.rfft(traces_aliased, axis=0) / lfp_filter[:, None]
        lfp_fourier -= fourier_aliased / np.exp(-2j * math.pi * lfp_freq[:, None] * shift_estimate)

        # Reconstruct using both AP and LFP channels
        # TODO: Have some flexibility on the ratio
        lfp_filt = self.lfp_filter(ap_freq)
        ratio = np.abs(lfp_filt[1:]) / (np.abs(lfp_filt[1:]) + np.abs(ap_filter[1:]))
        ratio = 1 / (1 + np.exp(-6 * np.tan(math.pi * (ratio - 0.5))))
        ratio = ratio[:, None]

        fourier_reconstructed = np.empty(ap_fourier.shape, dtype=np.complex128)
        fourier_reconstructed[nyquist_index:] = ap_fourier[nyquist_index:]
        fourier_reconstructed[:nyquist_index] = self.AP_TO_LFP * lfp_fourier * ratio[:nyquist_index] + ap_fourier[
            :nyquist_index
        ] * (1 - ratio[:nyquist_index])

        # To get back to the 0.5 - 10,000 Hz original filter
        # filter_reconstructed = generate_RC_filter(ap_freq, [0.5, 10000])[:, None]
        # fourier_reconstructed *= filter_reconstructed

        reconstructed_traces = np.fft.irfft(fourier_reconstructed, axis=0)

        if right_margin == 0:
            right_margin = -reconstructed_traces.shape[0]

        reconstructed_traces = reconstructed_traces[left_margin:-right_margin]

        return reconstructed_traces.astype(self.dtype)


class MergeNeuropixels1Recording(MergeApLfpRecording):
    """ """

    def __init__(self, ap_recording: BaseRecording, lfp_recording: BaseRecording, margin: int = 6_000) -> None:
        ap_filter = lambda f: generate_RC_filter(f, [300, 10000])
        lfp_filter = lambda f: generate_RC_filter(f, [0.5, 500])
        MergeApLfpRecording.__init__(self, ap_recording, lfp_recording, ap_filter, lfp_filter, margin)


def generate_RC_filter(frequencies: np.ndarray, cut: Union[float, List[float]], btype: str = "bandpass") -> np.ndarray:
    """
    Generates the transfer function of a single pole RC filter.

    Parameters
    ----------
    frequencies: np.ndarray
        The frequencies (in Hz) for which to generate the transfer function.
    cut: float | list[float]
        The cutoff frequency/frequencies (in Hz).
        Should be a float for lowpass/highpass and a list of 2 floats for bandpass.
    btype: str
        The type of filter. In "lowpass", "highpass", "bandpass".

    Returns
    -------
    transfer_function: np.ndarray
        The transfer function of the filter for each frequencies.
    """

    highpass = np.ones(len(frequencies), dtype=np.complex128)
    lowpass = np.ones(len(frequencies), dtype=np.complex128)

    if btype == "lowpass":
        lowpass = 1 / (1 + 1j * frequencies / cut)
    elif btype == "highpass":
        highpass = (1j * frequencies / cut) / (1 + 1j * frequencies / cut)
    elif btype == "bandpass":
        highpass = generate_RC_filter(frequencies, cut[0], btype="highpass")
        lowpass = generate_RC_filter(frequencies, cut[1], btype="lowpass")
    else:
        raise AttributeError(f"btype '{btype}' is invalid for generate_RC_filter.")

    return lowpass * highpass


def _time_shift_error(delay: float, ap_fft: np.ndarray, lfp_fft: np.ndarray, freq: np.ndarray) -> float:
    """
    Computes the error for a given delay between ap and lfp traces.

    Parameters
    ----------
    delay: float
        The delay (in s) between AP and LFP.
        Positive values indicate that lfp comes after ap.
    ap_fft: np.ndarray (n_freq, n_channels)
        The AP trace in the Fourier domain after unfiltering.
    lfp_fft: np.ndarray (n_freq, n_channels)
        The LFP trace in the Fourier domain after unfiltering.
    freq: np.ndarray (n_freq)
        The frequencies (in Hz).

    Returns
    -------
    error: float
        The error computed for the given delay.
    """

    expected_phase = -2 * math.pi * freq[:, None] * delay
    errors = np.angle(lfp_fft / ap_fft / np.exp(1j * expected_phase))

    return np.sum(np.abs(errors))
