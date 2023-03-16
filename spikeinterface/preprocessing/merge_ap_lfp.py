from typing import List, Union
import numpy as np

from ..core import BaseRecording, BaseRecordingSegment


class MergeApLfpRecording(BaseRecording):
    """
    Add cool description here.

    Parameters
    ----------
    ap_recording: BaseRecording
        The recording of the AP channels.
    lfp_recording: BaseRecording
        The recording of the LFP channels.

    Returns
    --------
    merged_ap_lfp_recording: MergeApLfpRecording
        The result of the merge of both channels (with the whole frequency spectrum).
    """

    def __init__(self, ap_recording: BaseRecording, lfp_recording: BaseRecording) -> None:
        BaseRecording.__init__(self, ap_recording.sampling_frequency, ap_recording.channel_ids, ap_recording.dtype)
        ap_recording.copy_metadata(self)

        for segment_index in range(ap_recording.get_num_segments()):
            recording_segment = MergeApLfpRecordingSegment(ap_recording._recording_segments[segment_index], lfp_recording._recording_segments[segment_index])
            self.add_recording_segment(recording_segment)

        self._kwargs = {
            'ap_recording': ap_recording.to_dict(),
            'lfp_recording': lfp_recording.to_dict()
        }


class MergeApLfpRecordingSegment(BaseRecordingSegment):

    def __init__(self, ap_recording_segment: BaseRecordingSegment, lfp_recording_segment: BaseRecordingSegment) -> None:
        self.ap_recording  = ap_recording_segment
        self.lfp_recording = lfp_recording_segment


    def get_traces(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None) -> np.ndarray:
        # TODO
        return self.ap_recording.get_traces(start_frame, end_frame, channel_indices)


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
