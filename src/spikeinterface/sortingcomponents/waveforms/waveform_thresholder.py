from pathlib import Path
import json
from typing import List, Optional
import scipy.signal

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import (
    PipelineNode,
    WaveformExtractorNode,
)


class WaveformThresholder(PipelineNode):

    def __init__(
        self, recording: BaseRecording, return_output: bool = True, 
        parents: Optional[List[PipelineNode]] = None,
        feature = 'ptp',
        threshold = 2
    ):
        super().__init__(recording, return_output=return_output, parents=parents)
        self.threshold = threshold
        self.feature = feature
        #if self.parents is None or not (len(self.parents) == 1 and isinstance(self.parents[0], WaveformExtractorNode)):
        #    exception_string = f"WaveformThresholder should have a single {WaveformExtractorNode.__name__} in its parents"
        #    raise TypeError(exception_string)
        
        self.nbefore = None

        ## We need to look for the last WaveformExtractor in the parents, 
        ## in order to get the size of the snippets
        parent = self.parents[-1]
        while self.nbefore is None:
            if hasattr(parent, 'nbefore'):
                self.nbefore = parent.nbefore
            if parent.parents is not None:
                parent = parent.parents[-1]
            else:
                break
        
    def compute(self, traces, peaks, waveforms):
        
        if self.feature == 'ptp':
            wf_data = waveforms.ptp(axis=1)
        elif self.feature == 'mean':
            wf_data = waveforms.mean(axis=1)
        elif self.feature == 'energy':
            wf_data = np.linalg.norm(waveforms, axis=1)
        elif self.feature == 'v_peak':
            wf_data = waveforms[idx][:, self.nbefore, :]

        mask = wf_data < self.threshold
        mask = np.repeat(mask[:, np.newaxis, :], waveforms.shape[1], axis=1)
        waveforms[mask] = 0

        return waveforms