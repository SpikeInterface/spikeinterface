from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path

import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

import random, string
from spikeinterface.core import get_global_tmp_folder
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.waveform_tools import estimate_templates
from .clustering_tools import remove_duplicates_via_matching
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
from sklearn.decomposition import TruncatedSVD
from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import compute_sparsity
from spikeinterface.sortingcomponents.tools import remove_empty_templates
import pickle, json
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
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
            "min_samples": 1,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "eom",
        },
        "cleaning_kwargs": {},
        "waveforms": {"ms_before": 2, "ms_after": 2},
        "sparsity": {"method": "ptp", "threshold": 0.25},
        "radius_um": 100,
        "n_svd": [5, 10],
        "ms_before": 0.5,
        "ms_after": 0.5,
        "noise_levels": None,
        "tmp_folder": None,
        "job_kwargs": {},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "random projections clustering needs hdbscan to be installed"

        job_kwargs = fix_job_kwargs(params["job_kwargs"])

        d = params
        verbose = job_kwargs.get("verbose", False)

        fs = recording.get_sampling_frequency()
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)

        if params["tmp_folder"] is None:
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / name
        else:
            tmp_folder = Path(params["tmp_folder"]).absolute()

        tmp_folder.mkdir(parents=True, exist_ok=True)

        # SVD for time compression
        few_peaks = select_peaks(peaks, method="uniform", n_peaks=10000)
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
            if all_pc_data.shape[1] > params["n_svd"][1]:
                tsvd = TruncatedSVD(params["n_svd"][1])
            else:
                tsvd = TruncatedSVD(all_pc_data.shape[1])
            sub_data = all_pc_data[mask]
            hdbscan_data = tsvd.fit_transform(sub_data.reshape(len(sub_data), -1))
            try:
                clustering = hdbscan.hdbscan(hdbscan_data, **d["hdbscan_kwargs"])
                local_labels = clustering[0]
            except Exception:
                local_labels = np.zeros(len(hdbscan_data))
            valid_clusters = local_labels > -1
            if np.sum(valid_clusters) > 0:
                local_labels[valid_clusters] += nb_clusters
                peak_labels[mask] = local_labels
                nb_clusters += len(np.unique(local_labels[valid_clusters]))

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        spikes = np.zeros(np.sum(peak_labels > -1), dtype=minimum_spike_dtype)
        mask = peak_labels > -1
        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        unit_ids = np.arange(len(np.unique(spikes["unit_index"])))

        nbefore = int(params["waveforms"]["ms_before"] * fs / 1000.0)
        nafter = int(params["waveforms"]["ms_after"] * fs / 1000.0)

        templates_array = estimate_templates(
            recording, spikes, unit_ids, nbefore, nafter, return_scaled=False, job_name=None, **job_kwargs
        )

        templates = Templates(
            templates_array, fs, nbefore, None, recording.channel_ids, unit_ids, recording.get_probe()
        )
        if params["noise_levels"] is None:
            params["noise_levels"] = get_noise_levels(recording, return_scaled=False)
        sparsity = compute_sparsity(templates, params["noise_levels"], **params["sparsity"])
        templates = templates.to_sparse(sparsity)
        templates = remove_empty_templates(templates)

        if verbose:
            print("We found %d raw clusters, starting to clean with matching..." % (len(templates.unit_ids)))

        cleaning_matching_params = params["job_kwargs"].copy()
        for value in ["chunk_size", "chunk_memory", "total_memory", "chunk_duration"]:
            if value in cleaning_matching_params:
                cleaning_matching_params.pop(value)
        cleaning_matching_params["chunk_duration"] = "100ms"
        cleaning_matching_params["n_jobs"] = 1
        cleaning_matching_params["verbose"] = False
        cleaning_matching_params["progress_bar"] = False

        cleaning_params = params["cleaning_kwargs"].copy()

        labels, peak_labels = remove_duplicates_via_matching(
            templates, peak_labels, job_kwargs=cleaning_matching_params, **cleaning_params
        )

        if verbose:
            print("We kept %d non-duplicated clusters..." % len(labels))

        return labels, peak_labels
