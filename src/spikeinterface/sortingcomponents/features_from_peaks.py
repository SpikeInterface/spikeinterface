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


class PeakToPeakLagsFeature(PipelineNode):
    def __init__(
        self,
        recording,
        name="ptp_lag_feature",
        return_output=True,
        parents=None,
        radius_um=150.0,
        all_channels=True,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.all_channels = all_channels
        self.radius_um = radius_um

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self._kwargs.update(dict(radius_um=radius_um, all_channels=all_channels))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        if self.all_channels:
            all_maxs = np.argmax(waveforms, axis=1)
            all_mins = np.argmin(waveforms, axis=1)
            all_lags = all_maxs - all_mins
        else:
            all_lags = np.zeros(peaks.size)
            for main_chan in np.unique(peaks["channel_index"]):
                (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
                (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
                wfs = waveforms[idx][:, :, chan_inds]
                maxs = np.argmax(wfs, axis=1)
                mins = np.argmin(wfs, axis=1)
                lags = maxs - mins
                ptps = np.argmax(np.ptp(wfs, axis=1), axis=1)
                all_lags[idx] = lags[np.arange(len(idx)), ptps]
        return all_lags


class RandomProjectionsFeature(PipelineNode):
    def __init__(
        self,
        recording,
        name="random_projections_feature",
        return_output=True,
        parents=None,
        projections=None,
        sigmoid=None,
        radius_um=None,
        sparse=True,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.projections = projections
        self.sigmoid = sigmoid
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self.radius_um = radius_um
        self.sparse = sparse
        self._kwargs.update(dict(projections=projections, sigmoid=sigmoid, radius_um=radius_um, sparse=sparse))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def _sigmoid(self, x):
        L, x0, k, b = self.sigmoid
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y

    def compute(self, traces, peaks, waveforms):
        all_projections = np.zeros((peaks.size, self.projections.shape[1]), dtype=self._dtype)

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            local_projections = self.projections[chan_inds, :]
            if self.sparse:
                wf_ptp = np.ptp(waveforms[idx][:, :, : len(chan_inds)], axis=1)
            else:
                wf_ptp = np.ptp(waveforms[idx][:, :, chan_inds], axis=1)

            if self.sigmoid is not None:
                wf_ptp *= self._sigmoid(wf_ptp)

            denom = np.sum(wf_ptp, axis=1)
            mask = denom != 0
            all_projections[idx[mask]] = np.dot(wf_ptp[mask], local_projections) / (denom[mask][:, np.newaxis])

        return all_projections


class RandomProjectionsEnergyFeature(PipelineNode):
    def __init__(
        self,
        recording,
        name="random_projections_energy_feature",
        return_output=True,
        parents=None,
        projections=None,
        radius_um=150.0,
        min_values=None,
    ):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self.projections = projections
        self.min_values = min_values
        self.radius_um = radius_um
        self._kwargs.update(dict(projections=projections, min_values=min_values, radius_um=radius_um))
        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        all_projections = np.zeros((peaks.size, self.projections.shape[1]), dtype=self._dtype)
        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            local_projections = self.projections[chan_inds, :]
            energies = np.linalg.norm(waveforms[idx][:, :, chan_inds], axis=1)

            if self.min_values is not None:
                energies = (energies / self.min_values[chan_inds]) ** 4

            denom = np.sum(energies, axis=1)
            mask = denom != 0

            all_projections[idx[mask]] = np.dot(energies[mask], local_projections) / (denom[mask][:, np.newaxis])
        return all_projections


class StdPeakToPeakFeature(PipelineNode):
    def __init__(self, recording, name="std_ptp_feature", return_output=True, parents=None, radius_um=150.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self._kwargs.update(dict(radius_um=radius_um))

        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        all_ptps = np.zeros(peaks.size)
        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx][:, :, chan_inds]
            all_ptps[idx] = np.std(np.ptp(wfs, axis=1), axis=1)
        return all_ptps


class GlobalPeakToPeakFeature(PipelineNode):
    def __init__(self, recording, name="global_ptp_feature", return_output=True, parents=None, radius_um=150.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self._kwargs.update(dict(radius_um=radius_um))

        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        all_ptps = np.zeros(peaks.size)
        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx][:, :, chan_inds]
            all_ptps[idx] = np.max(wfs, axis=(1, 2)) - np.min(wfs, axis=(1, 2))
        return all_ptps


class KurtosisPeakToPeakFeature(PipelineNode):
    def __init__(self, recording, name="kurtosis_ptp_feature", return_output=True, parents=None, radius_um=150.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self._kwargs.update(dict(radius_um=radius_um))

        self._dtype = recording.get_dtype()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks, waveforms):
        all_ptps = np.zeros(peaks.size)
        import scipy

        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])
            wfs = waveforms[idx][:, :, chan_inds]
            all_ptps[idx] = scipy.stats.kurtosis(np.ptp(wfs, axis=1), axis=1)
        return all_ptps


class EnergyFeature(PipelineNode):
    def __init__(self, recording, name="energy_feature", return_output=True, parents=None, radius_um=50.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

        self._kwargs.update(dict(radius_um=radius_um))

    def get_dtype(self):
        return np.dtype("float32")

    def compute(self, traces, peaks, waveforms):
        energy = np.zeros(peaks.size, dtype="float32")
        for main_chan in np.unique(peaks["channel_index"]):
            (idx,) = np.nonzero(peaks["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])

            wfs = waveforms[idx][:, :, chan_inds]
            energy[idx] = np.linalg.norm(wfs, axis=(1, 2)) / chan_inds.size
        return energy


_features_class = {
    "amplitude": AmplitudeFeature,
    "ptp": PeakToPeakFeature,
    "center_of_mass": LocalizeCenterOfMass,
    "monopolar_triangulation": LocalizeMonopolarTriangulation,
    "energy": EnergyFeature,
    "std_ptp": StdPeakToPeakFeature,
    "kurtosis_ptp": KurtosisPeakToPeakFeature,
    "random_projections_ptp": RandomProjectionsFeature,
    "random_projections_energy": RandomProjectionsEnergyFeature,
    "ptp_lag": PeakToPeakLagsFeature,
    "global_ptp": GlobalPeakToPeakFeature,
}
