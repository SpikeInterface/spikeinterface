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

from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.waveform_tools import estimate_templates
from .clustering_tools import remove_duplicates_via_matching
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser
from spikeinterface.sortingcomponents.features_from_peaks import RandomProjectionsFeature
from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import compute_sparsity
from spikeinterface.sortingcomponents.tools import remove_empty_templates
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
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
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
            "cluster_selection_epsilon": 2,
        },
        "cleaning_kwargs": {},
        "waveforms": {"ms_before": 2, "ms_after": 2},
        "sparsity": {"method": "ptp", "threshold": 0.25},
        "radius_um": 100,
        "nb_projections": 10,
        "feature": "energy",
        "ms_before": 0.5,
        "ms_after": 0.5,
        "random_seed": 42,
        "noise_levels": None,
        "smoothing_kwargs": {"window_length_ms": 0.25},
        "tmp_folder": None,
        "job_kwargs": {},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "random projections clustering need hdbscan to be installed"

        job_kwargs = fix_job_kwargs(params["job_kwargs"])

        d = params
        verbose = job_kwargs.get("verbose", False)

        fs = recording.get_sampling_frequency()
        radius_um = params["radius_um"]
        nbefore = int(params["ms_before"] * fs / 1000.0)
        nafter = int(params["ms_after"] * fs / 1000.0)
        num_chans = recording.get_num_channels()
        rng = np.random.RandomState(d["random_seed"])

        node0 = PeakRetriever(recording, peaks)
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=params["ms_before"],
            ms_after=params["ms_after"],
            radius_um=radius_um,
        )

        node2 = SavGolDenoiser(recording, parents=[node0, node1], return_output=False, **params["smoothing_kwargs"])

        num_projections = min(num_chans, d["nb_projections"])
        projections = rng.randn(num_chans, num_projections)
        if num_chans > 1:
            projections -= projections.mean()
            projections /= projections.std()

        nbefore = int(params["ms_before"] * fs / 1000)
        nafter = int(params["ms_after"] * fs / 1000)

        # if params["feature"] == "ptp":
        #     noise_values = np.ptp(rng.randn(1000, nsamples), axis=1)
        # elif params["feature"] == "energy":
        #     noise_values = np.linalg.norm(rng.randn(1000, nsamples), axis=1)
        # noise_threshold = np.mean(noise_values) + 3 * np.std(noise_values)

        node3 = RandomProjectionsFeature(
            recording,
            parents=[node0, node2],
            return_output=True,
            feature=params["feature"],
            projections=projections,
            radius_um=radius_um,
            noise_threshold=None,
            sparse=True,
        )

        pipeline_nodes = [node0, node1, node2, node3]

        hdbscan_data = run_node_pipeline(
            recording, pipeline_nodes, job_kwargs=job_kwargs, job_name="extracting features"
        )

        clustering = hdbscan.hdbscan(hdbscan_data, **d["hdbscan_kwargs"])
        peak_labels = clustering[0]

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

        cleaning_matching_params = job_kwargs.copy()
        for value in ["chunk_size", "chunk_memory", "total_memory", "chunk_duration"]:
            if value in cleaning_matching_params:
                cleaning_matching_params[value] = None
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
