from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from spikeinterface.core import Templates, estimate_templates, fix_job_kwargs
from spikeinterface.core.base import minimum_spike_dtype

# TODO find a way to attach a a sparse_mask to a given features (waveforms, pca, tsvd ....)


class FeaturesLoader:
    """
    Feature can be computed in memory or in a folder containing npy files.

    This class read the folder and behave like a dict of array lazily.

    Parameters
    ----------
    feature_folder

    preload

    """

    def __init__(self, feature_folder, preload=["peaks"]):
        self.feature_folder = Path(feature_folder)

        self.file_feature = {}
        self.loaded_features = {}
        for file in self.feature_folder.glob("*.npy"):
            name = file.stem
            if name in preload:
                self.loaded_features[name] = np.load(file)
            else:
                self.file_feature[name] = file

    def __getitem__(self, name):
        if name in self.loaded_features:
            return self.loaded_features[name]
        else:
            return np.load(self.file_feature[name], mmap_mode="r")

    @staticmethod
    def from_dict_or_folder(features_dict_or_folder):
        if isinstance(features_dict_or_folder, dict):
            return features_dict_or_folder
        else:
            return FeaturesLoader(features_dict_or_folder)


def aggregate_sparse_features(peaks, peak_indices, sparse_feature, sparse_target_mask, target_channels):
    """
    Aggregate sparse features that have unaligned channels and realigned then on target_channels.

    This is useful to aligned back peaks waveform or pca or tsvd when detected a differents channels.


    Parameters
    ----------
    peaks

    peak_indices

    sparse_feature

    sparse_target_mask

    target_channels

    Returns
    -------
    aligned_features: numpy.array
        Aligned features. shape is (local_peaks.size, sparse_feature.shape[1], target_channels.size)
    dont_have_channels: numpy.array
        Boolean vector to indicate spikes that do not have all target channels to be taken in account
        shape is peak_indices.size
    """
    local_peaks = peaks[peak_indices]

    aligned_features = np.zeros(
        (local_peaks.size, sparse_feature.shape[1], target_channels.size), dtype=sparse_feature.dtype
    )
    dont_have_channels = np.zeros(peak_indices.size, dtype=bool)

    for chan in np.unique(local_peaks["channel_index"]):
        sparse_chans = np.flatnonzero(sparse_target_mask[chan, :])
        peak_inds = np.flatnonzero(local_peaks["channel_index"] == chan)
        if np.all(np.isin(target_channels, sparse_chans)):
            # peaks feature channel have all target_channels
            source_chans = np.flatnonzero(np.isin(sparse_chans, target_channels))
            aligned_features[peak_inds, :, :] = sparse_feature[peak_indices[peak_inds], :, :][:, :, source_chans]
        else:
            # some channel are missing, peak are not removde
            dont_have_channels[peak_inds] = True

    return aligned_features, dont_have_channels


def apply_waveforms_shift(waveforms, peak_shifts, inplace=False):
    """
    Apply a shift a spike level to realign waveforms buffers.

    This is useful to compute template after merge when to cluster are shifted.

    A negative shift need the waveforms to be moved toward the right because the trough was too early.
    A positive shift need the waveforms to be moved toward the left because the trough was too late.

    Note the border sample are left as before without move.

    Parameters
    ----------

    waveforms

    peak_shifts

    inplace

    Returns
    -------
    aligned_waveforms


    """

    if inplace:
        aligned_waveforms = waveforms
    else:
        aligned_waveforms = waveforms.copy()

    shift_set = np.unique(peak_shifts)
    assert max(np.abs(shift_set)) < aligned_waveforms.shape[1]

    for shift in shift_set:
        if shift == 0:
            continue
        mask = peak_shifts == shift
        wfs = waveforms[mask]

        if shift > 0:
            aligned_waveforms[mask, :-shift, :] = wfs[:, shift:, :]
        else:
            aligned_waveforms[mask, -shift:, :] = wfs[:, :-shift, :]

    return aligned_waveforms


def get_templates_from_peaks_and_recording(
    recording,
    peaks,
    peak_labels,
    ms_before,
    ms_after,
    operator="average",
    job_kwargs=None,
):
    """
    Get templates from recording using the estimate_templates function.
    This function is a wrapper around the estimate_templates function from
    spikeinterface.core.waveform_tools. The function returns the Templates

    Parameters
    ----------
    recording : RecordingExtractor
        The recording object containing the data.
    peaks : numpy.ndarray
        The peaks array containing the peak information.
    peak_labels : numpy.ndarray
        The labels for each peak.
    ms_before : float
        The time window before the peak in milliseconds.
    ms_after : float
        The time window after the peak in milliseconds.
    operator : str
        The operator to use for template estimation. Can be 'average' or 'median'.
    job_kwargs : dict
        Additional keyword arguments for the estimate_templates function.

    Returns
    -------
    templates : Templates
        The estimated templates object.
    """

    job_kwargs = fix_job_kwargs(job_kwargs)

    mask = peak_labels > -1
    valid_peaks = peaks[mask]
    valid_labels = peak_labels[mask]
    labels, indices = np.unique(valid_labels, return_inverse=True)

    fs = recording.get_sampling_frequency()
    nbefore = int(ms_before * fs / 1000.0)
    nafter = int(ms_after * fs / 1000.0)

    spikes = np.zeros(valid_peaks.size, dtype=minimum_spike_dtype)
    spikes["sample_index"] = valid_peaks["sample_index"]
    spikes["unit_index"] = indices
    spikes["segment_index"] = valid_peaks["segment_index"]

    templates_array = estimate_templates(
        recording,
        spikes,
        np.arange(len(labels)),
        nbefore,
        nafter,
        operator=operator,
        return_in_uV=False,
        job_name=None,
        **job_kwargs,
    )

    templates = Templates(
        templates_array=templates_array,
        sampling_frequency=fs,
        nbefore=nbefore,
        sparsity_mask=None,
        channel_ids=recording.channel_ids,
        unit_ids=labels,
        probe=recording.get_probe(),
        is_in_uV=False,
    )

    return templates


def get_templates_from_peaks_and_svd(
    recording,
    peaks,
    peak_labels,
    ms_before,
    ms_after,
    svd_model,
    svd_features,
    sparsity_mask,
    operator="average",
    return_max_std_per_channel=False,
):
    """
    Get templates from recording using the SVD components

    Parameters
    ----------
    recording: RecordingExtractor
        The recording object containing the data.
    peaks : numpy.ndarray
        The peaks array containing the peak information.
    peak_labels : numpy.ndarray
        The labels for each peak.
    ms_before : float
        The time window before the peak in milliseconds.
    ms_after : float
        The time window after the peak in milliseconds.
    svd_model : TruncatedSVD
        The SVD model used to transform the data.
    svd_features : numpy.ndarray
        The SVD features array.
    sparsity_mask : numpy.ndarray
        The sparsity mask array.
    operator : str
        The operator to use for template estimation. Can be 'average' or 'median'.
    return_max_std_per_channel : bool
        Whether to return the max standard deviation at the channels.

    Returns
    -------
    dense_templates : Templates
        The estimated templates object as a dense template (but internanally contain sparse channels).
    final_sparsity_mask: np.array
        The final sparsity mask. Note that the template object is dense but with zeros.
    max_std_per_channel: np.array
        The maximal standard deviation of the templates per channel (only if return_max_std_per_channel is True).
    """

    assert operator in ["average", "median"], "operator should be either 'average' or 'median'"
    mask = peak_labels > -1
    valid_peaks = peaks[mask]
    valid_labels = peak_labels[mask]
    valid_svd_features = svd_features[mask]
    labels = np.unique(valid_labels)

    fs = recording.get_sampling_frequency()
    nbefore = int(ms_before * fs / 1000.0)
    nafter = int(ms_after * fs / 1000.0)
    num_channels = recording.get_num_channels()

    templates_array = np.zeros((len(labels), nbefore + nafter, num_channels), dtype=np.float32)
    if return_max_std_per_channel:
        max_std_per_channel = np.zeros((len(labels), num_channels), dtype=np.float32)

    final_sparsity_mask = np.zeros((len(labels), num_channels), dtype="bool")
    for unit_ind, label in enumerate(labels):
        mask = valid_labels == label
        local_peaks = valid_peaks[mask]
        local_svd = valid_svd_features[mask]
        peak_channels, b = np.unique(local_peaks["channel_index"], return_counts=True)
        best_channel = peak_channels[np.argmax(b)]
        sub_mask = local_peaks["channel_index"] == best_channel
        final_sparsity_mask[unit_ind, :] = sparsity_mask[best_channel]
        for count, i in enumerate(np.flatnonzero(sparsity_mask[best_channel])):
            if operator == "average":
                data = np.mean(local_svd[sub_mask, :, count], 0)
            elif operator == "median":
                data = np.median(local_svd[sub_mask, :, count], 0)
            templates_array[unit_ind, :, i] = svd_model.inverse_transform(data.reshape(1, -1))

            if return_max_std_per_channel:
                data = svd_model.inverse_transform(local_svd[sub_mask, :, count])
                if len(data) > 1:
                    max_std_per_channel[unit_ind, i] = np.std(data, 0).max()

    dense_templates = Templates(
        templates_array=templates_array,
        sampling_frequency=fs,
        nbefore=nbefore,
        sparsity_mask=None,
        channel_ids=recording.channel_ids,
        unit_ids=labels,
        probe=recording.get_probe(),
        is_in_uV=False,
    )

    if return_max_std_per_channel:
        return dense_templates, final_sparsity_mask, max_std_per_channel
    else:
        return dense_templates, final_sparsity_mask


def remove_small_cluster(recording, peaks, peak_labels, min_firing_rate=0.1, subsampling_factor=None, verbose=False):
    """
    Remove clusters too small in size (spike count) given a min firing rate and a subsampling factor.
    """

    if subsampling_factor is None:
        if verbose:
            print("remove_small_cluster(): subsampling_factor is not set, assuming 1")
        subsampling_factor = 1

    min_spike_count = int(recording.get_total_duration() * min_firing_rate / subsampling_factor)

    peak_labels = peak_labels.copy()
    labels_set, count = np.unique(peak_labels, return_counts=True)
    cluster_mask = count < min_spike_count
    to_remove = labels_set[cluster_mask]
    to_keep = labels_set[~cluster_mask]
    peak_mask = np.isin(peak_labels, to_remove)
    peak_labels[peak_mask] = -1

    to_keep = to_keep[to_keep >= 0]

    if verbose:
        print(
            f"remove_small_cluster: kept  {to_keep.size} removed {to_remove.size} (min_spike_count {min_spike_count})"
        )

    return peak_labels, to_keep
