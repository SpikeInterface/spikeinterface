from pathlib import Path
from typing import Any
import numpy as np


class FeaturesLoader:
    """
    Feature can be computed in memory or in a folder contaning npy files.

    This class read the folder and behave like a dict of array lazily.
    """
    def __init__(self, feature_folder, preload=['peaks']):
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



def aggregate_sparse_features(peaks, peak_indices, sparse_feature, sparse_mask, target_channels):
    """
    Aggregate sparse features that have unaligned channels and realigned then on target_channels


    """
    local_peaks = peaks[peak_indices]

    aligned_features = np.zeros(
        (local_peaks.size, sparse_feature.shape[1], target_channels.size), dtype=sparse_feature.dtype
    )
    dont_have_channels = np.zeros(peak_indices.size, dtype=bool)

    for chan in np.unique(local_peaks["channel_index"]):
        sparse_chans = np.flatnonzero(sparse_mask[chan, :])
        peak_inds = np.flatnonzero(local_peaks["channel_index"] == chan)
        if np.all(np.in1d(target_channels, sparse_chans)):
            # peaks feature channel have all target_channels
            source_chans = np.flatnonzero(np.in1d(sparse_chans, target_channels))
            aligned_features[peak_inds, :, :] = sparse_feature[peak_indices[peak_inds], :, :][:, :, source_chans]
        else:
            # some channel are missing, peak are not removde
            dont_have_channels[peak_inds] = True

    return aligned_features, dont_have_channels

def compute_template(peaks, peak_indices, sparse_waveforms, sparse_mask, target_channels):

    wfs = aggregate_sparse_features(peaks, peak_indices, sparse_waveforms, sparse_mask, target_channels)

    np.mean(wfs, axis=0)

