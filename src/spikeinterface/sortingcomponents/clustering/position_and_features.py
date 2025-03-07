from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path

import shutil
import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

import random, string, os
from spikeinterface.core import get_global_tmp_folder, get_noise_levels
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching, remove_duplicates_via_dip
from spikeinterface.core import NumpySorting
from spikeinterface.core import estimate_templates_with_accumulator, Templates
from spikeinterface.sortingcomponents.features_from_peaks import compute_features_from_peaks


class PositionAndFeaturesClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "hdbscan_kwargs": {
            "min_cluster_size": 50,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
        },
        "cleaning_kwargs": {},
        "radius_um": 100,
        "max_spikes_per_unit": 200,
        "selection_method": "random",
        "ms_before": 1.5,
        "ms_after": 1.5,
        "cleaning_method": "dip",
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        from sklearn.preprocessing import QuantileTransformer

        assert HAVE_HDBSCAN, "twisted clustering needs hdbscan to be installed"

        d = params

        peak_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]

        fs = recording.get_sampling_frequency()
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)
        num_samples = nbefore + nafter

        position_method = d["peak_localization_kwargs"]["method"]

        features_list = [
            position_method,
            "ptp",
        ]
        features_params = {
            position_method: {"radius_um": params["radius_um"]},
            "ptp": {"all_channels": False, "radius_um": params["radius_um"]},
        }

        features_data = compute_features_from_peaks(
            recording, peaks, features_list, features_params, ms_before=1, ms_after=1, **job_kwargs
        )

        hdbscan_data = np.zeros((len(peaks), 3), dtype=np.float32)
        hdbscan_data[:, 0] = features_data[0]["x"]
        hdbscan_data[:, 1] = features_data[0]["y"]
        hdbscan_data[:, 2] = features_data[1]

        preprocessing = QuantileTransformer(output_distribution="uniform")
        hdbscan_data = preprocessing.fit_transform(hdbscan_data)

        clusterer = hdbscan.HDBSCAN(**d["hdbscan_kwargs"])
        clusterer.fit(X=hdbscan_data)
        peak_labels = clusterer.labels_

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]  #  Noisy samples are given the label -1 in hdbscan

        best_spikes = {}
        num_spikes = 0

        all_indices = np.arange(0, peak_labels.size)

        max_spikes = params["max_spikes_per_unit"]
        selection_method = params["selection_method"]

        import sklearn

        for unit_ind in labels:
            mask = peak_labels == unit_ind
            if selection_method == "closest_to_centroid":
                data = hdbscan_data[mask]
                centroid = np.median(data, axis=0)
                distances = sklearn.metrics.pairwise_distances(centroid[np.newaxis, :], data)[0]
                best_spikes[unit_ind] = all_indices[mask][np.argsort(distances)[:max_spikes]]
            elif selection_method == "random":
                best_spikes[unit_ind] = np.random.permutation(all_indices[mask])[:max_spikes]
            num_spikes += best_spikes[unit_ind].size

        spikes = np.zeros(num_spikes, dtype=peak_dtype)

        mask = np.zeros(0, dtype=np.int32)
        for unit_ind in labels:
            mask = np.concatenate((mask, best_spikes[unit_ind]))

        idx = np.argsort(mask)
        mask = mask[idx]

        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        cleaning_method = params["cleaning_method"]

        print("We found %d raw clusters, starting to clean with %s..." % (len(labels), cleaning_method))

        if cleaning_method == "cosine":
            num_chans = recording.get_num_channels()
            wfs_arrays = extract_waveforms_to_buffers(
                recording,
                spikes,
                labels,
                nbefore,
                nafter,
                mode="shared_memory",
                return_scaled=False,
                folder=None,
                dtype=recording.get_dtype(),
                sparsity_mask=None,
                copy=True,
                **job_kwargs,
            )

            noise_levels = get_noise_levels(recording, return_scaled=False, **job_kwargs)
            labels, peak_labels = remove_duplicates(
                wfs_arrays, noise_levels, peak_labels, num_samples, num_chans, **params["cleaning_kwargs"]
            )

        elif cleaning_method == "dip":
            wfs_arrays = {}
            for label in labels:
                mask = label == peak_labels
                wfs_arrays[label] = hdbscan_data[mask]

            labels, peak_labels = remove_duplicates_via_dip(wfs_arrays, peak_labels, **params["cleaning_kwargs"])

        elif cleaning_method == "matching":
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))

            sorting = NumpySorting.from_samples_and_labels(spikes["sample_index"], spikes["unit_index"], fs)

            nbefore = int(params["ms_before"] * fs / 1000.0)
            nafter = int(params["ms_after"] * fs / 1000.0)
            templates_array = estimate_templates_with_accumulator(
                recording,
                sorting.to_spike_vector(),
                sorting.unit_ids,
                nbefore,
                nafter,
                return_scaled=False,
                **job_kwargs,
            )
            templates = Templates(
                templates_array=templates_array,
                sampling_frequency=fs,
                nbefore=nbefore,
                sparsity_mask=None,
                probe=recording.get_probe(),
                is_scaled=False,
            )

            labels, peak_labels = remove_duplicates_via_matching(
                templates, peak_labels, job_kwargs=job_kwargs, **params["cleaning_kwargs"]
            )
            shutil.rmtree(tmp_folder)

        print("We kept %d non-duplicated clusters..." % len(labels))

        return labels, peak_labels
