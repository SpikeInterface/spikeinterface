from pathlib import Path
import json
from typing import List, Optional
import scipy.signal
import numpy as np
import operator
from typing import Literal

from spikeinterface.core import BaseRecording
from spikeinterface.sortingcomponents.peak_pipeline import (
    PipelineNode,
    WaveformExtractorNode,
)


class WaveformThresholder(WaveformExtractorNode):
    """
    A node that performs waveform thresholding based on a selected feature.

    This node allows you to perform adaptive masking by setting channels to 0
    that have a given feature below a certain threshold. The available features
    to consider are peak-to-peak amplitude ('ptp'), mean amplitude ('mean'),
    energy ('energy'), and voltage peak ('v_peak').

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object.
    return_output: bool, optional
        Whether to return output from this node (default True).
    parents: list of PipelineNodes, optional
        The parent nodes of this node (default None).
    feature: {'ptp', 'mean', 'energy', 'v_peak'}, optional
        The feature to be considered for thresholding (default 'ptp').
    threshold: float, optional
        The threshold value for the selected feature (default 2).
    operator: callable, optional
        Comparator to flag values that should be set to 0 (default less or equal)
    """

    def __init__(
        self, recording: BaseRecording, return_output: bool = True, 
        parents: Optional[List[PipelineNode]] = None,
        feature: Literal['ptp', 'mean', 'energy', 'v_peak'] = 'ptp',
        threshold: float = 2,
        operator: callable = operator.le
    ):
        try:
            waveform_extractor = next(parent for parent in parents if isinstance(parent, WaveformExtractorNode))
        except (StopIteration, TypeError):
            exception_string = f"{self.__name__} should have a {WaveformExtractorNode.__name__} in its parents"
            raise TypeError(exception_string)

        super().__init__(recording, waveform_extractor.ms_before, waveform_extractor.ms_after,
            return_output=return_output, parents=parents)
        assert feature in ['ptp', 'mean', 'energy', 'v_peak'], f'{feature} is not a valid feature'

        self.threshold = threshold
        self.feature = feature
        self.operator = operator

        self._kwargs.update(dict(feature=feature,
                                 threshold=threshold,
                                 operator=operator))
        
    def compute(self, traces, peaks, waveforms):
        
        if self.feature == 'ptp':
            wf_data = waveforms.ptp(axis=1)
        elif self.feature == 'mean':
            wf_data = waveforms.mean(axis=1)
        elif self.feature == 'energy':
            wf_data = np.linalg.norm(waveforms, axis=1)
        elif self.feature == 'v_peak':
            wf_data = waveforms[:, self.nbefore, :]

        mask = self.operator(wf_data, self.threshold)
        mask = np.broadcast_to(mask[:, np.newaxis, :], waveforms.shape)
        waveforms[mask] = 0

        return waveforms