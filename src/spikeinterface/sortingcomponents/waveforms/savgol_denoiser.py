from pathlib import Path
import json
from typing import List, Optional
import scipy.signal

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class SavGolDenoiser(WaveformsNode):
    """
    Waveform Denoiser based on a simple Savitzky-Golay filtering
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object.
    return_output: bool, optional
        Whether to return output from this node (default True).
    parents: list of PipelineNodes, optional
        The parent nodes of this node (default None).
    order: int, optional
        the order of the filter (default 3)
    window_length_ms: float, optional
        the temporal duration of the filter in ms (default 0.25)
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
        order: int = 3,
        window_length_ms: float = 0.25,
    ):
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"SavGolDenoiser should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        self.order = order
        waveforms_sampling_frequency = self.recording.get_sampling_frequency()
        self.window_length = int(window_length_ms * waveforms_sampling_frequency / 1000)

        self._kwargs.update(dict(order=order, window_length_ms=window_length_ms))

    def compute(self, traces, peaks, waveforms):
        # Denoise
        denoised_waveforms = scipy.signal.savgol_filter(waveforms, self.window_length, self.order, axis=1)

        return denoised_waveforms
