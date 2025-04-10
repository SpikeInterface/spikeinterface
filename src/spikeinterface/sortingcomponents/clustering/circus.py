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
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.core.template import Templates
from spikeinterface.core.sparsity import compute_sparsity
from spikeinterface.sortingcomponents.tools import remove_empty_templates
from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd


from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel


class CircusClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "hdbscan_kwargs": {
            "min_cluster_size": 20,
            "cluster_selection_epsilon": 0.5,
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
        },
        "cleaning_kwargs": {},
        "remove_mixtures": False,
        "waveforms": {"ms_before": 2, "ms_after": 2},
        "sparsity": {"method": "snr", "amplitude_mode": "peak_to_peak", "threshold": 0.25},
        "recursive_kwargs": {
            "recursive": True,
            "recursive_depth": 3,
            "returns_split_count": True,
        },
        "split_kwargs": {"projection_mode": "tsvd", "n_pca_features": 0.9},
        "radius_um": 100,
        "neighbors_radius_um": 50,
        "n_svd": 5,
        "few_waveforms": None,
        "ms_before": 0.5,
        "ms_after": 0.5,
        "noise_threshold": 4,
        "rank": 5,
        "templates_from_svd": False,
        "noise_levels": None,
        "tmp_folder": None,
        "verbose": True,
        "debug": False,
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        assert HAVE_HDBSCAN, "random projections clustering needs hdbscan to be installed"

        d = params
        verbose = d["verbose"]

        fs = recording.get_sampling_frequency()
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        radius_um = params["radius_um"]
        neighbors_radius_um = params["neighbors_radius_um"]
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        if params["tmp_folder"] is None:
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / name
        else:
            tmp_folder = Path(params["tmp_folder"]).absolute()

        tmp_folder.mkdir(parents=True, exist_ok=True)

        # SVD for time compression
        if params["few_waveforms"] is None:
            few_peaks = select_peaks(
                peaks, recording=recording, method="uniform", n_peaks=10000, margin=(nbefore, nafter)
            )
            few_wfs = extract_waveform_at_max_channel(
                recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
            )
            wfs = few_wfs[:, :, 0]
        else:
            offset = int(params["waveforms"]["ms_before"] * fs / 1000)
            wfs = params["few_waveforms"][:, offset - nbefore : offset + nafter]

        # Ensure all waveforms have a positive max
        wfs *= np.sign(wfs[:, nbefore])[:, np.newaxis]

        # Remove outliers
        valid = np.argmax(np.abs(wfs), axis=1) == nbefore
        wfs = wfs[valid]

        from sklearn.decomposition import TruncatedSVD

        svd_model = TruncatedSVD(params["n_svd"])
        svd_model.fit(wfs)
        features_folder = tmp_folder / "tsvd_features"
        features_folder.mkdir(exist_ok=True)

        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording,
            peaks,
            ms_before=ms_before,
            ms_after=ms_after,
            svd_model=svd_model,
            radius_um=radius_um,
            folder=features_folder,
            **job_kwargs,
        )

        neighbours_mask = get_channel_distances(recording) <= neighbors_radius_um

        if params["debug"]:
            np.save(features_folder / "sparse_mask.npy", sparse_mask)
            np.save(features_folder / "peaks.npy", peaks)

        original_labels = peaks["channel_index"]
        from spikeinterface.sortingcomponents.clustering.split import split_clusters

        split_kwargs = params["split_kwargs"].copy()
        split_kwargs["neighbours_mask"] = neighbours_mask
        split_kwargs["waveforms_sparse_mask"] = sparse_mask
        split_kwargs["min_size_split"] = 2 * params["hdbscan_kwargs"].get("min_cluster_size", 50)
        split_kwargs["clusterer_kwargs"] = params["hdbscan_kwargs"]

        if params["debug"]:
            debug_folder = tmp_folder / "split"
        else:
            debug_folder = None

        peak_labels, _ = split_clusters(
            original_labels,
            recording,
            {"peaks": peaks, "sparse_tsvd": peaks_svd},
            method="local_feature_clustering",
            method_kwargs=split_kwargs,
            debug_folder=debug_folder,
            **params["recursive_kwargs"],
            **job_kwargs,
        )

        if params["noise_levels"] is None:
            params["noise_levels"] = get_noise_levels(recording, return_scaled=False, **job_kwargs)

        if not params["templates_from_svd"]:
            from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_recording

            templates = get_templates_from_peaks_and_recording(
                recording,
                peaks,
                peak_labels,
                ms_before,
                ms_after,
                **job_kwargs,
            )
        else:
            from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd

            templates = get_templates_from_peaks_and_svd(
                recording,
                peaks,
                peak_labels,
                ms_before,
                ms_after,
                svd_model,
                peaks_svd,
                sparse_mask,
                operator="median",
            )

        templates_array = templates.templates_array
        best_channels = np.argmax(np.abs(templates_array[:, nbefore, :]), axis=1)
        peak_snrs = np.abs(templates_array[:, nbefore, :])
        best_snrs_ratio = (peak_snrs / params["noise_levels"])[np.arange(len(peak_snrs)), best_channels]
        old_unit_ids = templates.unit_ids.copy()
        valid_templates = best_snrs_ratio > params["noise_threshold"]

        mask = np.isin(peak_labels, old_unit_ids[~valid_templates])
        peak_labels[mask] = -1

        from spikeinterface.core.template import Templates

        templates = Templates(
            templates_array=templates_array[valid_templates],
            sampling_frequency=fs,
            nbefore=templates.nbefore,
            sparsity_mask=None,
            channel_ids=recording.channel_ids,
            unit_ids=templates.unit_ids[valid_templates],
            probe=recording.get_probe(),
            is_scaled=False,
        )

        if params["debug"]:
            templates_folder = tmp_folder / "dense_templates"
            templates.to_zarr(folder_path=templates_folder)

        sparsity = compute_sparsity(templates, noise_levels=params["noise_levels"], **params["sparsity"])
        templates = templates.to_sparse(sparsity)
        empty_templates = templates.sparsity_mask.sum(axis=1) == 0
        old_unit_ids = templates.unit_ids.copy()
        templates = remove_empty_templates(templates)

        mask = np.isin(peak_labels, old_unit_ids[empty_templates])
        peak_labels[mask] = -1

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        if params["remove_mixtures"]:
            if verbose:
                print("Found %d raw clusters, starting to clean with matching" % (len(templates.unit_ids)))

            cleaning_job_kwargs = job_kwargs.copy()
            cleaning_job_kwargs["progress_bar"] = False
            cleaning_params = params["cleaning_kwargs"].copy()

            labels, peak_labels = remove_duplicates_via_matching(
                templates, peak_labels, job_kwargs=cleaning_job_kwargs, **cleaning_params
            )

            if verbose:
                print("Kept %d non-duplicated clusters" % len(labels))
        else:
            if verbose:
                print("Kept %d raw clusters" % len(labels))

        return labels, peak_labels, svd_model, peaks_svd, sparse_mask
