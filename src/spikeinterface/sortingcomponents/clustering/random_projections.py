from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path

import importlib
import numpy as np

hdbscan_spec = importlib.util.find_spec("hdbscan")
if hdbscan_spec is not None:
    HAVE_HDBSCAN = True
    import hdbscan
else:
    HAVE_HDBSCAN = False

from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.waveform_tools import estimate_templates
from spikeinterface.sortingcomponents.clustering.merging_tools import merge_peak_labels_from_templates
from spikeinterface.sortingcomponents.waveforms.savgol_denoiser import SavGolDenoiser
from spikeinterface.sortingcomponents.waveforms.features_from_peaks import RandomProjectionsFeature
from spikeinterface.core.template import Templates
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
        "clusterer": {
            "min_cluster_size": 10,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "eom",
        },
        "waveforms": {"ms_before": 0.5, "ms_after": 1.5},
        "sparsity": {"method": "snr", "amplitude_mode": "peak_to_peak", "threshold": 0.25},
        "radius_um": 50,
        "nb_projections": 10,
        "feature": "ptp",
        "seed": 42,
        "smoothing": {"window_length_ms": 0.25},
        "merge_from_templates": dict(),
        "debug_folder": None,
        "verbose": True,
    }

    name = "random_projections"
    params_doc = """
    """

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        assert HAVE_HDBSCAN, "random projections clustering need hdbscan to be installed"

        fs = recording.get_sampling_frequency()
        radius_um = params.get("radius_um", 30)
        ms_before = params["waveforms"].get("ms_before", 0.5)
        ms_after = params["waveforms"].get("ms_before", 1.5)
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        verbose = params.get("verbose", True)
        num_chans = recording.get_num_channels()
        debug_folder = params.get("debug_folder", None)

        if debug_folder is not None:
            debug_folder = Path(debug_folder).absolute()
            debug_folder.mkdir(exist_ok=True)

        rng = np.random.default_rng(params["seed"])

        node0 = PeakRetriever(recording, peaks)
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
            radius_um=radius_um,
        )

        node2 = SavGolDenoiser(recording, parents=[node0, node1], return_output=False, **params["smoothing"])

        num_projections = min(num_chans, params["nb_projections"])
        projections = rng.normal(loc=0.0, scale=1.0 / np.sqrt(num_chans), size=(num_chans, num_projections))

        node3 = RandomProjectionsFeature(
            recording,
            parents=[node0, node2],
            return_output=True,
            feature=params["feature"],
            projections=projections,
            radius_um=radius_um,
            sparse=True,
        )

        pipeline_nodes = [node0, node1, node2, node3]

        hdbscan_data = run_node_pipeline(
            recording, pipeline_nodes, job_kwargs=job_kwargs, job_name="extracting features", verbose=verbose
        )

        clustering = hdbscan.hdbscan(hdbscan_data, **params["clusterer"])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        spikes = np.zeros(np.sum(peak_labels > -1), dtype=minimum_spike_dtype)
        mask = peak_labels > -1
        spikes["sample_index"] = peaks[mask]["sample_index"]
        spikes["segment_index"] = peaks[mask]["segment_index"]
        spikes["unit_index"] = peak_labels[mask]

        unit_ids = np.arange(len(np.unique(spikes["unit_index"])))

        templates_array = estimate_templates(
            recording,
            spikes,
            unit_ids,
            nbefore,
            nafter,
            return_in_uV=False,
            job_name=None,
            **job_kwargs,
        )

        if verbose:
            print("Kept %d raw clusters" % len(labels))

        if params["merge_from_templates"] is not None:
            peak_labels, merge_template_array, new_sparse_mask, new_unit_ids = merge_peak_labels_from_templates(
                peaks,
                peak_labels,
                unit_ids,
                templates_array,
                np.ones((len(unit_ids), num_chans), dtype=bool),
                **params["merge_from_templates"],
            )

            templates = Templates(
                templates_array=merge_template_array,
                sampling_frequency=recording.sampling_frequency,
                nbefore=nbefore,
                sparsity_mask=None,
                channel_ids=recording.channel_ids,
                unit_ids=new_unit_ids,
                probe=recording.get_probe(),
                is_in_uV=False,
            )

        labels = templates.unit_ids

        if debug_folder is not None:
            templates.to_zarr(folder_path=debug_folder / "dense_templates")

        if verbose:
            print("Kept %d non-duplicated clusters" % len(labels))

        return labels, peak_labels, dict()
