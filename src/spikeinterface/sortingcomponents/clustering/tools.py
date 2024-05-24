from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np


# TODO find a way to attach a a sparse_mask to a given features (waveforms, pca, tsvd ....)


class FeaturesLoader:
    """
    Feature can be computed in memory or in a folder contaning npy files.

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


def aggregate_sparse_features(peaks, peak_indices, sparse_feature, sparse_mask, target_channels):
    """
    Aggregate sparse features that have unaligned channels and realigned then on target_channels.

    This is usefull to aligned back peaks waveform or pca or tsvd when detected a differents channels.


    Parameters
    ----------
    peaks

    peak_indices

    sparse_feature

    sparse_mask

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
        sparse_chans = np.flatnonzero(sparse_mask[chan, :])
        peak_inds = np.flatnonzero(local_peaks["channel_index"] == chan)
        if np.all(np.isin(target_channels, sparse_chans)):
            # peaks feature channel have all target_channels
            source_chans = np.flatnonzero(np.in1d(sparse_chans, target_channels))
            aligned_features[peak_inds, :, :] = sparse_feature[peak_indices[peak_inds], :, :][:, :, source_chans]
        else:
            # some channel are missing, peak are not removde
            dont_have_channels[peak_inds] = True

    return aligned_features, dont_have_channels


def compute_template_from_sparse(
    peaks, labels, labels_set, sparse_waveforms, sparse_mask, total_channels, peak_shifts=None
):
    """
    Compute template average from single sparse waveforms buffer.

    Parameters
    ----------
    peaks

    labels

    labels_set

    sparse_waveforms

    sparse_mask

    total_channels

    peak_shifts

    Returns
    -------
    templates: numpy.array
        Templates shape : (len(labels_set), num_samples, total_channels)
    """
    n = len(labels_set)

    templates = np.zeros((n, sparse_waveforms.shape[1], total_channels), dtype=sparse_waveforms.dtype)

    for i, label in enumerate(labels_set):
        peak_indices = np.flatnonzero(labels == label)

        local_chans = np.unique(peaks["channel_index"][peak_indices])
        target_channels = np.flatnonzero(np.all(sparse_mask[local_chans, :], axis=0))

        aligned_wfs, dont_have_channels = aggregate_sparse_features(
            peaks, peak_indices, sparse_waveforms, sparse_mask, target_channels
        )

        if peak_shifts is not None:
            apply_waveforms_shift(aligned_wfs, peak_shifts[peak_indices], inplace=True)

        templates[i, :, :][:, target_channels] = np.mean(aligned_wfs[~dont_have_channels], axis=0)

    return templates


def apply_waveforms_shift(waveforms, peak_shifts, inplace=False):
    """
    Apply a shift a spike level to realign waveforms buffers.

    This is usefull to compute template after merge when to cluster are shifted.

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

    print("apply_waveforms_shift")

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

    print("apply_waveforms_shift DONE")

    return aligned_waveforms
