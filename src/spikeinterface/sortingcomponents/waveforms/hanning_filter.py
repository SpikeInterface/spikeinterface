from __future__ import annotations


from typing import List, Optional
import numpy as np
from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class HanningFilter(WaveformsNode):
    """
    Hanning Filtering to remove border effects while extracting waveforms

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
    ):
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"HanningFilter should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        hanning_before = np.hanning(2 * self.nbefore)
        hanning_after = np.hanning(2 * self.nafter)
        hanning = np.concatenate((hanning_before[: self.nbefore], hanning_after[self.nafter :]))
        self.hanning = hanning[:, None]
        self._kwargs.update(dict())

    def compute(self, traces, peaks, waveforms):
        denoised_waveforms = waveforms * self.hanning
        return denoised_waveforms
