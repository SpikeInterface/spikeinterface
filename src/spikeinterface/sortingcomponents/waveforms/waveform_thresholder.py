from pathlib import Path
import json
from typing import List, Optional
import scipy.signal
import numpy as np

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import (
    PipelineNode,
    WaveformExtractorNode,
)


class WaveformThresholder(WaveformExtractorNode):
    """
    Waveform Thresholder based on selected feature
    This allow to perform an adaptive masking, by setting to 0 channels
    that have a given feature below a given threshold

    Parameters
    ----------
    feature: string in ['ptp', 'mean', 'energy', 'v_peak']
        the feature that should be considered
    threshold: float
        the threshold below which channels are set to 0
    """

    def __init__(
        self, recording: BaseRecording, return_output: bool = True, 
        parents: Optional[List[PipelineNode]] = None,
        feature = 'ptp',
        threshold = 2
    ):
        try:
            waveform_extractor = next(parent for parent in parents if isinstance(parent, WaveformExtractorNode))
        except (StopIteration, TypeError):
            exception_string = f"Model should have a {WaveformExtractorNode.__name__} in its parents"
            raise TypeError(exception_string)

        super().__init__(recording, waveform_extractor.ms_before, waveform_extractor.ms_after,
            return_output=return_output, parents=parents)

        self.threshold = threshold
        self.feature = feature
        
    def compute(self, traces, peaks, waveforms):
        
        if self.feature == 'ptp':
            wf_data = waveforms.ptp(axis=1)
        elif self.feature == 'mean':
            wf_data = waveforms.mean(axis=1)
        elif self.feature == 'energy':
            wf_data = np.linalg.norm(waveforms, axis=1)
        elif self.feature == 'v_peak':
            wf_data = waveforms[:, self.nbefore, :]

        mask = wf_data < self.threshold
        mask = np.repeat(mask[:, np.newaxis, :], waveforms.shape[1], axis=1)
        waveforms[mask] = 0

        return waveforms