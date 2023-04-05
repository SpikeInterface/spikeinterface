from pathlib import Path
import json
from typing import List, Optional
import scipy.signal

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import PipelineNode, WaveformExtractorNode, find_parent_of_type


class SavGolDenoiser(PipelineNode):
    def __init__(
        self, recording: BaseRecording, return_output: bool = True, 
        parents: Optional[List[PipelineNode]] = None,
        order : int = 3,
        window_length_ms : float = 0.25
    ):
        super().__init__(recording, return_output=return_output, parents=parents)
        self.order = order

        waveform_extractor = find_parent_of_type(self, WaveformExtractorNode)
        if waveform_extractor is None:
            raise TypeError(f"SavGolDenoiser should have a single {WaveformExtractorNode.__name__} in its parents")
        waveforms_sampling_frequency = waveform_extractor.recording.get_sampling_frequency()
        self.window_length = int(window_length_ms*waveforms_sampling_frequency / 1000)

    def compute(self, traces, peaks, waveforms):
        # Denoise
        denoised_waveforms = scipy.signal.savgol_filter(waveforms, self.window_length, self.order, axis=1)

        return denoised_waveforms