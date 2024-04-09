"""Sorting components: peak waveform features."""

from __future__ import annotations


import numpy as np

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core import get_channel_distances
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass, LocalizeMonopolarTriangulation
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    PipelineNode,
    ExtractDenseWaveforms,
)


def compute_features_from_peaks(
    recording,
    peaks,
    feature_list=[
        "ptp",
    ],
    feature_params={},
    ms_before=1.0,
    ms_after=1.0,
    **job_kwargs,
):
    """Extract features on the fly from the recording given a list of peaks.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way
    feature_list: list, default: ["ptp"]
        List of features to be computed. Possible features are:
            - amplitude
            - ptp
            - center_of_mass
            - energy
            - std_ptp
            - ptp_lag
            - random_projections_ptp
            - random_projections_energy
    ms_before: float, default: 1.0
        The duration in ms before the peak for extracting the features
    ms_after: float, default: 1.0
        The duration in ms  after the peakfor extracting the features

    {}

    Returns
    -------
    A tuple of features. Even if there is one feature.
    Every feature have shape[0] == peaks.shape[0].
    dtype and other dim depends on features.

    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    peak_retriever = PeakRetriever(recording, peaks)
    extract_dense_waveforms = ExtractDenseWaveforms(
        recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=False
    )
    nodes = [
        peak_retriever,
        extract_dense_waveforms,
    ]
    for feature_name in feature_list:
        assert (
            feature_name in _features_class.keys()
        ), f"Feature {feature_name} in 'feature_list' is not possible. Possible features are {list(_features_class.keys())}"
        Class = _features_class[feature_name]
        params = feature_params.get(feature_name, {}).copy()
        node = Class(recording, parents=[peak_retriever, extract_dense_waveforms], **params)
        nodes.append(node)

    features = run_node_pipeline(recording, nodes, job_kwargs, job_name="features_from_peaks", squeeze_output=False)

    return features


class AmplitudeFeature(PipelineNode):
    def __init__(
        self, recording, name="amplitude_feature", return_output=True, parents=None, all_channels=False, peak_sign="neg"
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.all_channels = all_channels
        self.peak_sign = peak_sign
        self._kwargs.update(dict(all_channels=all_channels, peak_sign=peak_sign))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        if self.all_channels:
            if self.peak_sign == "neg":
                amplitudes = np.min(waveforms, axis=1)
            elif self.peak_sign == "pos":
                amplitudes = np.max(waveforms, axis=1)
            elif self.peak_sign == "both":
                amplitudes = np.max(np.abs(waveforms, axis=1))
        else:
            if self.peak_sign == "neg":
                amplitudes = np.min(waveforms, axis=(1, 2))
            elif self.peak_sign == "pos":
                amplitudes = np.max(waveforms, axis=(1, 2))
            elif self.peak_sign == "both":
                amplitudes = np.max(np.abs(waveforms), axis=(1, 2))
        return amplitudes


class PeakToPeakFeature(PipelineNode):
    def __init__(
        self, recording, name="ptp_feature", return_output=True, parents=None, radius_um=150.0, all_channels=True
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self.all_channels = all_channels
        self._kwargs.update(dict(radius_um=radius_um, all_channels=all_channels))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        if self.all_channels:
            all_ptps = np.ptp(waveforms, axis=1)
        else:
            all_ptps = np.zeros(peaks.size)
            for main_chan in np.unique(peaks["channel_index"]):
                (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
                (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                all_ptps[idx] = np.max(np.ptp(wfs, axis=1))
        return all_ptps


class RandomProjectionsFeature(PipelineNode):
    def __init__(
        self,
        recording,
        name="random_projections_feature",
        feature="ptp",
        return_output=True,
        parents=None,
        projections=None,
        radius_um=100,
        sparse=True,
        noise_threshold=None,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        assert feature in ["ptp", "energy"]
        self.projections = projections
        self.feature = feature
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self.radius_um = radius_um
        self.sparse = sparse
        self.noise_threshold = noise_threshold
        self._kwargs.update(
            dict(
                projections=projections,
                radius_um=radius_um,
                sparse=sparse,
                noise_threshold=noise_threshold,
                feature=feature,
            )
        )
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        all_projections = np.zeros((peaks.size, self.projections.shape[1]), dtype=self._dtype)

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            local_projections = self.projections[chan_inds, :]
            if self.sparse:
                if self.feature == "ptp":
                    features = np.ptp(waveforms[idx][:, :, : len(chan_inds)], axis=1)
                elif self.feature == "energy":
                    features = np.linalg.norm(waveforms[idx][:, :, : len(chan_inds)], axis=1)
            else:
                if self.feature == "ptp":
                    features = np.ptp(waveforms[idx][:, :, chan_inds], axis=1)
                elif self.feature == "energy":
                    features = np.linalg.norm(waveforms[idx][:, :, chan_inds], axis=1)

            if self.noise_threshold is not None:
                local_map = np.median(features, axis=0) < self.noise_threshold
                features[features < local_map] = 0

            all_projections[idx] = np.dot(features, local_projections)

        return all_projections


_features_class = {
    "amplitude": AmplitudeFeature,
    "ptp": PeakToPeakFeature,
    "random_projections": RandomProjectionsFeature,
    "center_of_mass": LocalizeCenterOfMass,
}
