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
from spikeinterface.core import get_global_tmp_folder, get_noise_levels, get_channel_distances, get_random_data_chunks
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching, remove_duplicates_via_dip
from spikeinterface.core import NumpySorting
from spikeinterface.core import extract_waveforms
from spikeinterface.sortingcomponents.features_from_peaks import compute_features_from_peaks, EnergyFeature


class RandomProjectionClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "hdbscan_kwargs": {
            "min_cluster_size": 20,
            "allow_single_cluster": True,
            "core_dist_n_jobs": os.cpu_count(),
            "cluster_selection_method": "leaf",
        },
        "cleaning_kwargs": {},
        "radius_um": 100,
        "max_spikes_per_unit": 200,
        "selection_method": "closest_to_centroid",
        "nb_projections": {"ptp": 8, "energy": 2},
        "ms_before": 1.5,
        "ms_after": 1.5,
        "random_seed": 42,
        "cleaning_method": "matching",
        "shared_memory": False,
        "min_values": {"ptp": 0, "energy": 0},
        "tmp_folder": None,
        "job_kwargs": {"n_jobs": os.cpu_count(), "chunk_memory": "10M", "verbose": True, "progress_bar": True},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "random projections clustering need hdbscan to be installed"

        if "n_jobs" in params["job_kwargs"]:
            if params["job_kwargs"]["n_jobs"] == -1:
                params["job_kwargs"]["n_jobs"] = os.cpu_count()

        if "core_dist_n_jobs" in params["hdbscan_kwargs"]:
            if params["hdbscan_kwargs"]["core_dist_n_jobs"] == -1:
                params["hdbscan_kwargs"]["core_dist_n_jobs"] = os.cpu_count()

        d = params
        verbose = d["job_kwargs"]["verbose"]

        peak_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]

        fs = recording.get_sampling_frequency()
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)
        num_samples = nbefore + nafter
        num_chans = recording.get_num_channels()

        noise_levels = get_noise_levels(recording, return_scaled=False)

        np.random.seed(d["random_seed"])

        features_params = {}
        features_list = []

        noise_snippets = None

        for proj_type in ["ptp", "energy"]:
            if d["nb_projections"][proj_type] > 0:
                features_list += [f"random_projections_{proj_type}"]

                if d["min_values"][proj_type] == "auto":
                    if noise_snippets is None:
                        num_segments = recording.get_num_segments()
                        num_chunks = 3 * d["max_spikes_per_unit"] // num_segments
                        noise_snippets = get_random_data_chunks(
                            recording, num_chunks_per_segment=num_chunks, chunk_size=num_samples, seed=42
                        )
                        noise_snippets = noise_snippets.reshape(num_chunks, num_samples, num_chans)

                    if proj_type == "energy":
                        data = np.linalg.norm(noise_snippets, axis=1)
                        min_values = np.median(data, axis=0)
                    elif proj_type == "ptp":
                        data = np.ptp(noise_snippets, axis=1)
                        min_values = np.median(data, axis=0)
                elif d["min_values"][proj_type] > 0:
                    min_values = d["min_values"][proj_type]
                else:
                    min_values = None

                projections = np.random.randn(num_chans, d["nb_projections"][proj_type])
                features_params[f"random_projections_{proj_type}"] = {
                    "radius_um": params["radius_um"],
                    "projections": projections,
                    "min_values": min_values,
                }

        features_data = compute_features_from_peaks(
            recording, peaks, features_list, features_params, ms_before=1, ms_after=1, **params["job_kwargs"]
        )

        if len(features_data) > 1:
            hdbscan_data = np.hstack((features_data[0], features_data[1]))
        else:
            hdbscan_data = features_data[0]

        import sklearn

        clustering = hdbscan.hdbscan(hdbscan_data, **d["hdbscan_kwargs"])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        best_spikes = {}
        nb_spikes = 0

        all_indices = np.arange(0, peak_labels.size)

        max_spikes = params["max_spikes_per_unit"]
        selection_method = params["selection_method"]

        for unit_ind in labels:
            mask = peak_labels == unit_ind
            if selection_method == "closest_to_centroid":
                data = hdbscan_data[mask]
                centroid = np.median(data, axis=0)
                distances = sklearn.metrics.pairwise_distances(centroid[np.newaxis, :], data)[0]
                best_spikes[unit_ind] = all_indices[mask][np.argsort(distances)[:max_spikes]]
            elif selection_method == "random":
                best_spikes[unit_ind] = np.random.permutation(all_indices[mask])[:max_spikes]
            nb_spikes += best_spikes[unit_ind].size

        spikes = np.zeros(nb_spikes, dtype=peak_dtype)

        mask = np.zeros(0, dtype=np.int32)
        for unit_ind in labels:
            mask = np.concatenate((mask, best_spikes[unit_ind]))

        idx = np.argsort(mask)
        mask = mask[idx]

        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        cleaning_method = params["cleaning_method"]

        if verbose:
            print("We found %d raw clusters, starting to clean with %s..." % (len(labels), cleaning_method))

        if cleaning_method == "cosine":
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
                **params["job_kwargs"],
            )

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
            # create a tmp folder
            if params["tmp_folder"] is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = get_global_tmp_folder() / name
            else:
                tmp_folder = Path(params["tmp_folder"])

            if params["shared_memory"]:
                waveform_folder = None
                mode = "memory"
            else:
                waveform_folder = tmp_folder / "waveforms"
                mode = "folder"

            sorting_folder = tmp_folder / "sorting"
            sorting = NumpySorting.from_times_labels(spikes["sample_index"], spikes["unit_index"], fs)
            sorting = sorting.save(folder=sorting_folder)
            we = extract_waveforms(
                recording,
                sorting,
                waveform_folder,
                ms_before=params["ms_before"],
                ms_after=params["ms_after"],
                **params["job_kwargs"],
                return_scaled=False,
                mode=mode,
            )

            cleaning_matching_params = params["job_kwargs"].copy()
            cleaning_matching_params["chunk_duration"] = "100ms"
            cleaning_matching_params["n_jobs"] = 1
            cleaning_matching_params["verbose"] = False
            cleaning_matching_params["progress_bar"] = False

            cleaning_params = params["cleaning_kwargs"].copy()
            cleaning_params["tmp_folder"] = tmp_folder

            labels, peak_labels = remove_duplicates_via_matching(
                we, noise_levels, peak_labels, job_kwargs=cleaning_matching_params, **cleaning_params
            )

            if params["tmp_folder"] is None:
                shutil.rmtree(tmp_folder)
            else:
                shutil.rmtree(tmp_folder / "waveforms")
                shutil.rmtree(tmp_folder / "sorting")

        if verbose:
            print("We kept %d non-duplicated clusters..." % len(labels))

        return labels, peak_labels
