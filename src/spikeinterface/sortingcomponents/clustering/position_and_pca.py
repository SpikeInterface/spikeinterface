from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path
import random
import string
import os

import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

from spikeinterface.core import get_global_tmp_folder
from spikeinterface.core.recording_tools import get_channel_distances, get_random_data_chunks
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import auto_clean_clustering, auto_split_clustering


class PositionAndPCAClustering:
    """
    Perform a hdbscan clustering on peak position then apply locals
    PCA on waveform + hdbscan on every spatial clustering to check
    if there a need to oversplit. Should be fairly close to spyking-circus
    clustering

    """

    _default_params = {
        "peak_locations": None,
        "use_amplitude": True,
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "ms_before": 1.5,
        "ms_after": 2.5,
        "n_components_by_channel": 3,
        "n_components": 5,
        "job_kwargs": {"n_jobs": -1, "chunk_memory": "10M", "progress_bar": True},
        "hdbscan_global_kwargs": {"min_cluster_size": 20, "allow_single_cluster": True, "core_dist_n_jobs": -1},
        "hdbscan_local_kwargs": {"min_cluster_size": 20, "allow_single_cluster": True, "core_dist_n_jobs": -1},
        "waveform_mode": "shared_memory",
        "radius_um": 50.0,
        "noise_size": 300,
        "debug": False,
        "tmp_folder": None,
        "auto_merge_num_shift": 3,
        "auto_merge_quantile_limit": 0.8,
        "ratio_num_channel_intersect": 0.5,
    }

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()

        tmp_folder = params["tmp_folder"]
        if params["waveform_mode"] == "memmap":
            if tmp_folder is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))
            else:
                tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir()
            params2["tmp_folder"] = tmp_folder
        elif params["waveform_mode"] == "shared_memory":
            assert tmp_folder is None, "tmp_folder must be None for shared_memory"
        else:
            raise ValueError("shared_memory")

        return params2

    @classmethod
    def main_function(cls, recording, peaks, params):
        # res = PositionClustering(recording, peaks, params)

        assert HAVE_HDBSCAN, "position_and_pca clustering need hdbscan to be installed"

        params = cls._check_params(recording, peaks, params)
        # wfs_arrays, sparsity_mask, noise = cls._initialize_folder(recording, peaks, params)

        # step1 : clustering on peak location
        if params["peak_locations"] is None:
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks

            peak_locations = localize_peaks(
                recording, peaks, **params["peak_localization_kwargs"], **params["job_kwargs"]
            )
        else:
            peak_locations = params["peak_locations"]

        location_keys = ["x", "y"]
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)

        if params["use_amplitude"]:
            to_cluster_from = np.hstack((locations, peaks["amplitude"][:, np.newaxis]))
        else:
            to_cluster_from = locations

        clusterer = hdbscan.HDBSCAN(**params["hdbscan_global_kwargs"])
        clusterer.fit(X=to_cluster_from)
        spatial_peak_labels = clusterer.labels_

        spatial_labels = np.unique(spatial_peak_labels)
        spatial_labels = spatial_labels[spatial_labels >= 0]  #  Noisy samples are given the label -1 in hdbscan

        # step2 : extract waveform by cluster
        (spatial_keep,) = np.nonzero(spatial_peak_labels >= 0)

        keep_peak_labels = spatial_peak_labels[spatial_keep]

        peak_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
        peaks2 = np.zeros(spatial_keep.size, dtype=peak_dtype)
        peaks2["sample_index"] = peaks["sample_index"][spatial_keep]
        peaks2["segment_index"] = peaks["segment_index"][spatial_keep]

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((spatial_labels.size, num_chans), dtype="bool")
        chan_locs = recording.get_channel_locations()
        chan_distances = get_channel_distances(recording)
        for l, label in enumerate(spatial_labels):
            mask = keep_peak_labels == label
            peaks2["unit_index"][mask] = l

            center = np.median(locations[spatial_keep][mask], axis=0)
            main_chan = np.argmin(np.linalg.norm(chan_locs - center[np.newaxis, :], axis=1))

            # TODO take a radius that depend on the cluster dispertion itself
            (closest_chans,) = np.nonzero(chan_distances[main_chan, :] <= params["radius_um"])
            sparsity_mask[l, closest_chans] = True

        if params["waveform_mode"] == "shared_memory":
            wf_folder = None
        else:
            assert params["tmp_folder"] is not None
            wf_folder = params["tmp_folder"] / "sparse_snippets"
            wf_folder.mkdir()

        fs = recording.get_sampling_frequency()
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)

        ids = np.arange(num_chans, dtype="int64")
        wfs_arrays = extract_waveforms_to_buffers(
            recording,
            peaks2,
            spatial_labels,
            nbefore,
            nafter,
            mode=params["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=recording.get_dtype(),
            sparsity_mask=sparsity_mask,
            copy=(params["waveform_mode"] == "shared_memory"),
            **params["job_kwargs"],
        )

        noise = get_random_data_chunks(
            recording,
            return_scaled=False,
            num_chunks_per_segment=params["noise_size"],
            chunk_size=nbefore + nafter,
            concatenated=False,
            seed=None,
        )
        noise = np.stack(noise, axis=0)

        print("Launching the local pca for splitting purposes")
        split_peak_labels, main_channels = auto_split_clustering(
            wfs_arrays,
            sparsity_mask,
            spatial_labels,
            keep_peak_labels,
            nbefore,
            nafter,
            noise,
            n_components_by_channel=params["n_components_by_channel"],
            n_components=params["n_components"],
            hdbscan_params=params["hdbscan_local_kwargs"],
            debug=params["debug"],
            debug_folder=params["tmp_folder"],
        )

        peak_labels = -2 * np.ones(peaks.size, dtype=np.int64)
        peak_labels[spatial_keep] = split_peak_labels

        # auto clean
        pre_clean_labels = np.unique(peak_labels)
        pre_clean_labels = pre_clean_labels[pre_clean_labels >= 0]
        # ~ print('labels before auto clean', pre_clean_labels.size, pre_clean_labels)

        peaks3 = np.zeros(peaks.size, dtype=peak_dtype)
        peaks3["sample_index"] = peaks["sample_index"]
        peaks3["segment_index"] = peaks["segment_index"]
        peaks3["unit_index"][:] = -1
        sparsity_mask3 = np.zeros((pre_clean_labels.size, num_chans), dtype="bool")
        for l, label in enumerate(pre_clean_labels):
            peaks3["unit_index"][peak_labels == label] = l
            main_chan = main_channels[label]
            (closest_chans,) = np.nonzero(chan_distances[main_chan, :] <= params["radius_um"])
            sparsity_mask3[l, closest_chans] = True

        if params["waveform_mode"] == "shared_memory":
            wf_folder = None
        else:
            if params["tmp_folder"] is not None:
                wf_folder = params["tmp_folder"] / "waveforms_pre_autoclean"
                wf_folder.mkdir()

        wfs_arrays3 = extract_waveforms_to_buffers(
            recording,
            peaks3,
            pre_clean_labels,
            nbefore,
            nafter,
            mode=params["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=recording.get_dtype(),
            sparsity_mask=sparsity_mask3,
            copy=(params["waveform_mode"] == "shared_memory"),
            **params["job_kwargs"],
        )

        clean_peak_labels, peak_sample_shifts = auto_clean_clustering(
            wfs_arrays3,
            sparsity_mask3,
            pre_clean_labels,
            peak_labels,
            nbefore,
            nafter,
            chan_distances,
            radius_um=params["radius_um"],
            auto_merge_num_shift=params["auto_merge_num_shift"],
            auto_merge_quantile_limit=params["auto_merge_quantile_limit"],
            ratio_num_channel_intersect=params["ratio_num_channel_intersect"],
        )

        # final
        labels = np.unique(clean_peak_labels)
        labels = labels[labels >= 0]

        return labels, clean_peak_labels
