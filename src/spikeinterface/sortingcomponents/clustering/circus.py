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
from spikeinterface.core import get_global_tmp_folder, get_channel_distances
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching, remove_duplicates_via_dip
from spikeinterface.core import NumpySorting
from spikeinterface.core import extract_waveforms
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
from sklearn.decomposition import TruncatedSVD
import pickle, json
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    PeakRetriever,
)
from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel


class CircusClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "hdbscan_kwargs": {
            "min_cluster_size": 20,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "eom",
        },
        "cleaning_kwargs": {},
        "waveforms": {"ms_before": 2, "ms_after": 2, "max_spikes_per_unit": 100},
        "radius_um": 100,
        "selection_method": "closest_to_centroid",
        "n_svd": [5, 10],
        "ms_before": 1,
        "ms_after": 1,
        "random_seed": 42,
        "shared_memory": True,
        "tmp_folder": None,
        "job_kwargs": {"n_jobs": os.cpu_count(), "chunk_memory": "100M", "verbose": True, "progress_bar": True},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "random projections clustering needs hdbscan to be installed"

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
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        num_samples = nbefore + nafter
        num_chans = recording.get_num_channels()
        np.random.seed(d["random_seed"])

        if params["tmp_folder"] is None:
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / name
        else:
            tmp_folder = Path(params["tmp_folder"]).absolute()

        tmp_folder.mkdir(parents=True, exist_ok=True)

        # SVD for time compression
        few_peaks = select_peaks(peaks, method="uniform", n_peaks=5000)
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **params["job_kwargs"]
        )

        wfs = few_wfs[:, :, 0]
        tsvd = TruncatedSVD(params["n_svd"][0])
        tsvd.fit(wfs)

        model_folder = tmp_folder / "tsvd_model"

        model_folder.mkdir(exist_ok=True)
        with open(model_folder / "pca_model.pkl", "wb") as f:
            pickle.dump(tsvd, f)

        model_params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": float(fs),
        }

        with open(model_folder / "params.json", "w") as f:
            json.dump(model_params, f)

        # features
        features_folder = model_folder / "features"
        node0 = PeakRetriever(recording, peaks)

        radius_um = params["radius_um"]
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
            radius_um=radius_um,
        )

        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, model_folder_path=model_folder
        )

        pipeline_nodes = [node0, node1, node2]

        all_pc_data = run_node_pipeline(
            recording,
            pipeline_nodes,
            params["job_kwargs"],
            job_name="extracting features",
        )

        peak_labels = -1 * np.ones(len(peaks), dtype=int)
        nb_clusters = 0
        for c in np.unique(peaks["channel_index"]):
            mask = peaks["channel_index"] == c
            tsvd = TruncatedSVD(params["n_svd"][1])
            sub_data = all_pc_data[mask]
            hdbscan_data = tsvd.fit_transform(sub_data.reshape(len(sub_data), -1))
            try:
                clustering = hdbscan.hdbscan(hdbscan_data, **d["hdbscan_kwargs"])
                local_labels = clustering[0]
            except Exception:
                local_labels = -1 * np.ones(len(hdbscan_data))
            valid_clusters = local_labels > -1
            if np.sum(valid_clusters) > 0:
                local_labels[valid_clusters] += nb_clusters
                peak_labels[mask] = local_labels
                nb_clusters += len(np.unique(local_labels[valid_clusters]))

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        best_spikes = {}
        nb_spikes = 0

        import sklearn

        all_indices = np.arange(0, peak_labels.size)

        max_spikes = params["waveforms"]["max_spikes_per_unit"]
        selection_method = params["selection_method"]

        for unit_ind in labels:
            mask = peak_labels == unit_ind
            if selection_method == "closest_to_centroid":
                data = all_pc_data[mask].reshape(np.sum(mask), -1)
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
            precompute_template=["median"],
            mode=mode,
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
