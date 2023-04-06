from pathlib import Path
import json
from typing import List, Optional
import scipy.signal

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import PipelineNode, WaveformExtractorNode


class SavGolDenoiser(WaveformExtractorNode):
    """
    Waveform Denoiser based on a simple Savitky Golay filtering

    Parameters
    ----------
    order: int 
        the order of the filter (default 3)
    window_length_ms: float
        the temporal duration of the filter in ms (default 0.25)
    """

    def __init__(
        self, recording: BaseRecording, return_output: bool = True, 
        parents: Optional[List[PipelineNode]] = None,
        order : int = 3,
        window_length_ms : float = 0.25
    ):
        try:
            waveform_extractor = next(parent for parent in parents if isinstance(parent, WaveformExtractorNode))
        except (StopIteration, TypeError):
            exception_string = f"Model should have a {WaveformExtractorNode.__name__} in its parents"
            raise TypeError(exception_string)

        super().__init__(recording, waveform_extractor.ms_before, waveform_extractor.ms_after,
            return_output=return_output, parents=parents)

        self.order = order
        waveforms_sampling_frequency = self.recording.get_sampling_frequency()
        self.window_length = int(window_length_ms*waveforms_sampling_frequency / 1000)

    def compute(self, traces, peaks, waveforms):
        # Denoise
        denoised_waveforms = scipy.signal.savgol_filter(waveforms, self.window_length, self.order, axis=1)

        return denoised_waveforms