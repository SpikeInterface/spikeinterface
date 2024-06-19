from __future__ import annotations

from typing import List, Optional
import numpy as np
import operator
from typing import Literal

from spikeinterface.core import BaseRecording, get_noise_levels
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class WaveformThresholder(WaveformsNode):
    """
    A node that performs waveform thresholding based on a selected feature.

    This node allows you to perform adaptive masking by setting channels to 0
    that have a given feature below a certain threshold. The available features
    to consider are peak-to-peak amplitude ("ptp"), mean amplitude ("mean"),
    energy ("energy"), and peak voltage ("peak_voltage").

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node
    feature: "ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        The feature to be considered for thresholding . Features are normalized with the channel noise levels.
    threshold: float, default: 2
        The threshold value for the selected feature
    noise_levels: array of None, default: None
        The noise levels to determine the thresholds
    random_chunk_kwargs: dict, default: dict()
        Parameters for computing noise levels, if not provided (sub optimal)
    operator: callable, default: operator.le (less or equal)
        Comparator to flag values that should be set to 0
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
        feature: Literal["ptp", "mean", "energy", "peak_voltage"] = "ptp",
        threshold: float = 2,
        noise_levels: Optional[np.array] = None,
        random_chunk_kwargs: dict = {},
        operator: callable = operator.le,
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
        assert feature in ["ptp", "mean", "energy", "peak_voltage"], (
            f"{feature} is not a valid feature" " must be one of 'ptp', 'mean', 'energy'," " or 'peak_voltage'"
        )

        self.threshold = threshold
        self.feature = feature
        self.operator = operator

        self.noise_levels = noise_levels
        if self.noise_levels is None:
            self.noise_levels = get_noise_levels(self.recording, **random_chunk_kwargs, return_scaled=False)

        self._kwargs.update(
            dict(feature=feature, threshold=threshold, operator=operator, noise_levels=self.noise_levels)
        )

    def compute(self, traces, peaks, waveforms):
        if self.feature == "ptp":
            wf_data = np.ptp(waveforms, axis=1) / self.noise_levels
        elif self.feature == "mean":
            wf_data = waveforms.mean(axis=1) / self.noise_levels
        elif self.feature == "energy":
            wf_data = np.linalg.norm(waveforms, axis=1) / self.noise_levels
        elif self.feature == "peak_voltage":
            wf_data = waveforms[:, self.nbefore, :] / self.noise_levels

        mask = self.operator(wf_data, self.threshold)
        mask = np.broadcast_to(mask[:, np.newaxis, :], waveforms.shape)
        waveforms[mask] = 0

        return waveforms
