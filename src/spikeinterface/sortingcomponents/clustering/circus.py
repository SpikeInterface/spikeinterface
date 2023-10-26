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
from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser
from spikeinterface.sortingcomponents.features_from_peaks import RandomProjectionsFeature, PeakToPeakFeature
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    PeakRetriever,
)


class CircusClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "hdbscan_kwargs": {
            "min_cluster_size": 20,
            "allow_single_cluster": True,
            "core_dist_n_jobs": os.cpu_count(),
            "cluster_selection_method": "eom",
        },
        "cleaning_kwargs": {},
        "waveforms": {"ms_before": 2, "ms_after": 2, "max_spikes_per_unit": 100},
        "radius_um": 100,
        "selection_method": "closest_to_centroid",
        "nb_projections": 10,
        "ms_before": 1,
        "ms_after": 1,
        "random_seed": 42,
        "noise_levels": None,
        "smoothing_kwargs": {"window_length_ms": 0.25},
        "shared_memory": True,
        "tmp_folder": None,
        "job_kwargs": {"n_jobs": os.cpu_count(), "chunk_memory": "100M", "verbose": True, "progress_bar": True},
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

        if d["noise_levels"] is None:
            noise_levels = get_noise_levels(recording, return_scaled=False)
        else:
            noise_levels = d["noise_levels"]

        np.random.seed(d["random_seed"])

        if params["tmp_folder"] is None:
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / name
        else:
            tmp_folder = Path(params["tmp_folder"]).absolute()

        tmp_folder.mkdir(parents=True, exist_ok=True)

        node0 = PeakRetriever(recording, peaks)
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=params["ms_before"],
            ms_after=params["ms_after"],
            radius_um=params["radius_um"],
        )

        node2 = SavGolDenoiser(recording, parents=[node0, node1], return_output=False, **params["smoothing_kwargs"])
        node3 = PeakToPeakFeature(recording, parents=[node0, node2])

        pipeline_nodes = [node0, node1, node2, node3]
        all_ptps = run_node_pipeline(
            recording, pipeline_nodes, params["job_kwargs"], job_name="extracting features"
        )

        import sklearn.decomposition
        import sklearn

        peak_labels = -1 * np.ones(len(peaks), dtype=int)
        nb_clusters = 0

        best_spikes = {}
        nb_spikes = 0
        max_components = 5

        max_spikes = params["waveforms"]["max_spikes_per_unit"]
        selection_method = params["selection_method"]

        for main_chan in np.unique(peaks['channel_index']):
            mask = peaks['channel_index'] == main_chan
            n_components = min(max_components, np.sum(mask))
            svd = sklearn.decomposition.TruncatedSVD(n_components)
            hdbscan_data = svd.fit_transform(all_ptps[mask])

            try:
                clustering = hdbscan.hdbscan(hdbscan_data, **d['hdbscan_kwargs'])
                local_labels = clustering[0]
                valid_clusters = local_labels > -1
                all_indices, = np.where(mask)
            except Exception:
                valid_clusters = np.zeros(0, dtype=bool)

            if np.sum(valid_clusters) > 0:
                local_labels[valid_clusters] += nb_clusters
                peak_labels[mask] = local_labels
                nb_clusters += len(np.unique(local_labels[valid_clusters]))

                for unit_ind in np.unique(local_labels[valid_clusters]):
                    sub_mask = local_labels == unit_ind
                    if selection_method == "closest_to_centroid":
                        data = hdbscan_data[sub_mask]
                        centroid = np.median(data, axis=0)
                        distances = sklearn.metrics.pairwise_distances(centroid[np.newaxis, :], data)[0]
                        best_spikes[unit_ind] = all_indices[sub_mask][np.argsort(distances)[:max_spikes]]
                    elif selection_method == "random":
                        best_spikes[unit_ind] = np.random.permutation(all_indices[sub_mask])[:max_spikes]
                    nb_spikes += best_spikes[unit_ind].size
        
        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        spikes = np.zeros(nb_spikes, dtype=peak_dtype)

        mask = np.zeros(0, dtype=np.int32)
        for unit_ind in labels:
            mask = np.concatenate((mask, best_spikes[unit_ind]))

        idx = np.argsort(mask)
        mask = mask[idx]

        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        if verbose:
            print("We found %d raw clusters, starting to clean with matching..." % (len(labels)))

        sorting_folder = tmp_folder / "sorting"
        unit_ids = np.arange(len(np.unique(spikes["unit_index"])))
        sorting = NumpySorting(spikes, fs, unit_ids=unit_ids)

        if params["shared_memory"]:
            waveform_folder = None
            mode = "memory"
        else:
            waveform_folder = tmp_folder / "waveforms"
            mode = "folder"
            sorting = sorting.save(folder=sorting_folder)

        we = extract_waveforms(
            recording,
            sorting,
            waveform_folder,
            **params["job_kwargs"],
            **params["waveforms"],
            return_scaled=False,
            mode=mode,
        )

        cleaning_matching_params = params["job_kwargs"].copy()
        for value in ["chunk_size", "chunk_memory", "total_memory", "chunk_duration"]:
            if value in cleaning_matching_params:
                cleaning_matching_params.pop(value)
        cleaning_matching_params["chunk_duration"] = "100ms"
        cleaning_matching_params["n_jobs"] = 1
        cleaning_matching_params["verbose"] = False
        cleaning_matching_params["progress_bar"] = False

        cleaning_params = params["cleaning_kwargs"].copy()
        cleaning_params["tmp_folder"] = tmp_folder

        labels, peak_labels = remove_duplicates_via_matching(
            we, noise_levels, peak_labels, job_kwargs=cleaning_matching_params, **cleaning_params
        )

        del we, sorting

        if params["tmp_folder"] is None:
            shutil.rmtree(tmp_folder)
        else:
            if not params["shared_memory"]:
                shutil.rmtree(tmp_folder / "waveforms")
                shutil.rmtree(tmp_folder / "sorting")

        if verbose:
            print("We kept %d non-duplicated clusters..." % len(labels))

        return labels, peak_labels
