from typing import Callable, List, Union
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
    margin: int
        The margin (in samples) to use when extracting the trace.
        Takes the frequencies as parameter, and outputs the transfer function.

    Returns
    --------
    merged_ap_lfp_recording: MergeApLfpRecording
        The result of the merge of both channels (with the whole frequency spectrum).
    """

    def __init__(self, ap_recording: BaseRecording, lfp_recording: BaseRecording, ap_filter: Callable[[np.ndarray], np.ndarray],
                 lfp_filter: Callable[[np.ndarray], np.ndarray], margin: int = 1000) -> None:
        BaseRecording.__init__(self, ap_recording.sampling_frequency, ap_recording.channel_ids, ap_recording.dtype)
        ap_recording.copy_metadata(self)

        for segment_index in range(ap_recording.get_num_segments()):
            ap_recording_segment  = ap_recording._recording_segments[segment_index]
            lfp_recording_segment = lfp_recording._recording_segments[segment_index]
            self.add_recording_segment(MergeApLfpRecordingSegment(ap_recording_segment, lfp_recording_segment, ap_filter, lfp_filter, margin))

        self._kwargs = {  # TODO: Is callable serializable?
            'ap_recording': ap_recording.to_dict(),
            'lfp_recording': lfp_recording.to_dict(),
            'margin': margin
        }


class MergeApLfpRecordingSegment(BaseRecordingSegment):

    def __init__(self, ap_recording_segment: BaseRecordingSegment, lfp_recording_segment: BaseRecordingSegment,
                 ap_filter: Callable[[np.ndarray], np.ndarray], lfp_filter: Callable[[np.ndarray], np.ndarray], margin: int) -> None:
        self.ap_recording  = ap_recording_segment
        self.lfp_recording = lfp_recording_segment
        self.ap_filter  = ap_filter
        self.lfp_filter = lfp_filter
        self.margin = margin


    def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None) -> np.ndarray:
        ap_traces, left_margin_ap, right_margin_ap    = get_chunk_with_margin(self.ap_recording, start_frame, end_frame, channel_indices, self.margin)
        lfp_traces, left_margin_lfp, right_margin_lfp = get_chunk_with_margin(self.lfp_recording, start_frame, end_frame, channel_indices, self.margin)

        ap_fourier  = np.fft.rfft(ap_traces, axis=0)
        lfp_fourier = np.fft.rfft(lfp_traces, axis=0)
        ap_freq =  np.fft.rfftfreq(ap_traces.shape[0],  d=1/self.ap_recording.sampling_frequency)
        lfp_freq = np.fft.rfftfreq(lfp_traces.shape[0], d=1/self.lfp_recording.sampling_frequency)

        ap_filter  = self.ap_filter(ap_freq)
        lfp_filter = self.lfp_filter(lfp_freq)
        ap_filter  = np.where(ap_filter  == 0, 1.0, ap_filter)
        lfp_filter = np.where(lfp_filter == 0, 1.0, lfp_filter)

        reconstructed_ap_fourier  = ap_fourier  /  ap_filter[:, None]
        reconstructed_lfp_fourier = lfp_fourier / lfp_filter[:, None]

        # TODO: LFP anti-aliasing
        # TODO: reconstruct using both files

        reconstructed_traces = np.fft.irfft(reconstructed_ap_fourier, axis=0)  # TODO: is a placeholder

        if right_margin_ap == 0:
            right_margin_ap = -reconstructed_traces.shape[0]

        reconstructed_traces = reconstructed_traces[left_margin_ap : -right_margin_ap]

        return reconstructed_traces.astype(self.ap_recording.dtype)


class MergeNeuropixels1Recording(MergeApLfpRecording):
    """

    """

    def __init__(self, ap_recording: BaseRecording, lfp_recording: BaseRecording, margin: int = 1000) -> None:
        ap_filter  = lambda f : generate_RC_filter(f, [300, 10000])
        lfp_filter = lambda f : generate_RC_filter(f, [0.5, 500])
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
    lowpass  = np.ones(len(frequencies), dtype=np.complex128)

    if btype == "lowpass":
        lowpass = 1 / (1 + 1j * frequencies / cut)
    elif btype == "highpass":
        highpass = (frequencies / cut) / (1 + 1j * frequencies / cut)
    elif btype == "bandpass":
        highpass = generate_RC_filter(frequencies, cut[0], btype="highpass")
        lowpass  = generate_RC_filter(frequencies, cut[1], btype="lowpass")
    else:
        raise AttributeError(f"btype '{btype}' is invalid for generate_RC_filter.")

    return lowpass * highpass
