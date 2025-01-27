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
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
from spikeinterface.sortingcomponents.waveforms.hanning_filter import HanningFilter
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
            "cluster_selection_epsilon": 0.5,
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
        },
        "cleaning_kwargs": {},
        "waveforms": {"ms_before": 2, "ms_after": 2},
        "sparsity": {"method": "snr", "amplitude_mode": "peak_to_peak", "threshold": 0.25},
        "recursive_kwargs": {
            "recursive": True,
            "recursive_depth": 3,
            "returns_split_count": True,
        },
        "radius_um": 100,
        "n_svd": 5,
        "few_waveforms": None,
        "ms_before": 0.5,
        "ms_after": 0.5,
        "noise_threshold": 4,
        "rank": 5,
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

        # Perform Hanning filtering
        hanning_before = np.hanning(2 * nbefore)
        hanning_after = np.hanning(2 * nafter)
        hanning = np.concatenate((hanning_before[:nbefore], hanning_after[nafter:]))
        wfs *= hanning

        from sklearn.decomposition import TruncatedSVD

        tsvd = TruncatedSVD(params["n_svd"])
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

        node2 = HanningFilter(recording, parents=[node0, node1], return_output=False)

        node3 = TemporalPCAProjection(
            recording, parents=[node0, node2], return_output=True, model_folder_path=model_folder
        )

        pipeline_nodes = [node0, node1, node2, node3]

        if len(params["recursive_kwargs"]) == 0:
            from sklearn.decomposition import PCA

            all_pc_data = run_node_pipeline(
                recording,
                pipeline_nodes,
                job_kwargs,
                job_name="extracting features",
            )

            peak_labels = -1 * np.ones(len(peaks), dtype=int)
            nb_clusters = 0
            for c in np.unique(peaks["channel_index"]):
                mask = peaks["channel_index"] == c
                sub_data = all_pc_data[mask]
                sub_data = sub_data.reshape(len(sub_data), -1)

                if all_pc_data.shape[1] > params["n_svd"]:
                    tsvd = PCA(params["n_svd"], whiten=True)
                else:
                    tsvd = PCA(all_pc_data.shape[1], whiten=True)

                hdbscan_data = tsvd.fit_transform(sub_data)
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
        else:

            features_folder = tmp_folder / "tsvd_features"
            features_folder.mkdir(exist_ok=True)

            _ = run_node_pipeline(
                recording,
                pipeline_nodes,
                job_kwargs,
                job_name="extracting features",
                gather_mode="npy",
                gather_kwargs=dict(exist_ok=True),
                folder=features_folder,
                names=["sparse_tsvd"],
            )

            sparse_mask = node1.neighbours_mask
            neighbours_mask = get_channel_distances(recording) <= radius_um

            # np.save(features_folder / "sparse_mask.npy", sparse_mask)
            np.save(features_folder / "peaks.npy", peaks)

            original_labels = peaks["channel_index"]
            from spikeinterface.sortingcomponents.clustering.split import split_clusters

            min_size = 2 * params["hdbscan_kwargs"].get("min_cluster_size", 20)

            if params["debug"]:
                debug_folder = tmp_folder / "split"
            else:
                debug_folder = None

            peak_labels, _ = split_clusters(
                original_labels,
                recording,
                features_folder,
                method="local_feature_clustering",
                method_kwargs=dict(
                    clusterer="hdbscan",
                    feature_name="sparse_tsvd",
                    neighbours_mask=neighbours_mask,
                    waveforms_sparse_mask=sparse_mask,
                    min_size_split=min_size,
                    clusterer_kwargs=d["hdbscan_kwargs"],
                    n_pca_features=5,
                ),
                debug_folder=debug_folder,
                **params["recursive_kwargs"],
                **job_kwargs,
            )

        non_noise = peak_labels > -1
        labels, inverse = np.unique(peak_labels[non_noise], return_inverse=True)
        peak_labels[non_noise] = inverse
        labels = np.unique(inverse)

        spikes = np.zeros(non_noise.sum(), dtype=minimum_spike_dtype)
        spikes["sample_index"] = peaks[non_noise]["sample_index"]
        spikes["segment_index"] = peaks[non_noise]["segment_index"]
        spikes["unit_index"] = peak_labels[non_noise]

        unit_ids = labels

        nbefore = int(params["waveforms"]["ms_before"] * fs / 1000.0)
        nafter = int(params["waveforms"]["ms_after"] * fs / 1000.0)

        if params["noise_levels"] is None:
            params["noise_levels"] = get_noise_levels(recording, return_scaled=False, **job_kwargs)

        templates_array = estimate_templates(
            recording,
            spikes,
            unit_ids,
            nbefore,
            nafter,
            return_scaled=False,
            job_name=None,
            **job_kwargs,
        )

        best_channels = np.argmax(np.abs(templates_array[:, nbefore, :]), axis=1)
        peak_snrs = np.abs(templates_array[:, nbefore, :])
        best_snrs_ratio = (peak_snrs / params["noise_levels"])[np.arange(len(peak_snrs)), best_channels]
        valid_templates = best_snrs_ratio > params["noise_threshold"]

        if d["rank"] is not None:
            from spikeinterface.sortingcomponents.matching.circus import compress_templates

            _, _, _, templates_array = compress_templates(templates_array, d["rank"])

        templates = Templates(
            templates_array=templates_array[valid_templates],
            sampling_frequency=fs,
            nbefore=nbefore,
            sparsity_mask=None,
            channel_ids=recording.channel_ids,
            unit_ids=unit_ids[valid_templates],
            probe=recording.get_probe(),
            is_scaled=False,
        )

        sparsity = compute_sparsity(templates, noise_levels=params["noise_levels"], **params["sparsity"])
        templates = templates.to_sparse(sparsity)
        empty_templates = templates.sparsity_mask.sum(axis=1) == 0
        templates = remove_empty_templates(templates)

        mask = np.isin(peak_labels, np.where(empty_templates)[0])
        peak_labels[mask] = -1

        mask = np.isin(peak_labels, np.where(~valid_templates)[0])
        peak_labels[mask] = -1

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

        return labels, peak_labels
