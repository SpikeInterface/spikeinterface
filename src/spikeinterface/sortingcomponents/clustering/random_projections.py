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
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core import get_global_tmp_folder, get_channel_distances, get_random_data_chunks
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching, remove_duplicates_via_dip
from spikeinterface.core import NumpySorting
from spikeinterface.core import extract_waveforms
from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser
from spikeinterface.sortingcomponents.features_from_peaks import RandomProjectionsFeature
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    PeakRetriever,
)


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
        "waveforms": {"ms_before": 2, "ms_after": 2, "max_spikes_per_unit": 100},
        "radius_um": 100,
        "selection_method": "closest_to_centroid",
        "nb_projections": 10,
        "ms_before": 1,
        "ms_after": 1,
        "random_seed": 42,
        "smoothing_kwargs": {"window_length_ms": 0.25},
        "shared_memory": True,
        "tmp_folder": None,
        "debug": False,
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

        fs = recording.get_sampling_frequency()
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)
        num_samples = nbefore + nafter
        num_chans = recording.get_num_channels()
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

        num_projections = min(num_chans, d["nb_projections"])
        projections = np.random.randn(num_chans, num_projections)
        if num_chans > 1:
            projections -= projections.mean(0)
            projections /= projections.std(0)

        nbefore = int(params["ms_before"] * fs / 1000)
        nafter = int(params["ms_after"] * fs / 1000)
        nsamples = nbefore + nafter

        node3 = RandomProjectionsFeature(
            recording,
            parents=[node0, node2],
            return_output=True,
            projections=projections,
            radius_um=params["radius_um"],
            sparse=True,
        )

        pipeline_nodes = [node0, node1, node2, node3]

        hdbscan_data = run_node_pipeline(
            recording, pipeline_nodes, params["job_kwargs"], job_name="extracting features"
        )

        import sklearn

        clustering = hdbscan.hdbscan(hdbscan_data, **d["hdbscan_kwargs"])
        peak_labels = clustering[0]

        # peak_labels = -1 * np.ones(len(peaks), dtype=int)
        # nb_clusters = 0
        # for c in np.unique(peaks['channel_index']):
        #     mask = peaks['channel_index'] == c
        #     clustering = hdbscan.hdbscan(hdbscan_data[mask], **d['hdbscan_kwargs'])
        #     local_labels = clustering[0]
        #     valid_clusters = local_labels > -1
        #     if np.sum(valid_clusters) > 0:
        #         local_labels[valid_clusters] += nb_clusters
        #         peak_labels[mask] = local_labels
        #         nb_clusters += len(np.unique(local_labels[valid_clusters]))

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        best_spikes = {}
        nb_spikes = 0

        all_indices = np.arange(0, peak_labels.size)

        max_spikes = params["waveforms"]["max_spikes_per_unit"]
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

        spikes = np.zeros(nb_spikes, dtype=minimum_spike_dtype)

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
            return_scaled=False,
            mode=mode,
            precompute_template=["median"],
            **params["job_kwargs"],
            **params["waveforms"],
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
            we, peak_labels, job_kwargs=cleaning_matching_params, **cleaning_params
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
