from .si_based import ComponentsBasedSorter

from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels, extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore

import numpy as np


import pickle
import json


class SimpleSorter(ComponentsBasedSorter):
    """
    Implementation of a very simple sorter usefull for teaching.
    The idea is quite old school:
      * detect peaks
      * project waveforms with SVD or PCA
      * apply a well known clustering algos from scikit-learn

      No template matching. No auto cleaning.

      Mainly usefull for few channels (1 to 8), teaching and testing.
    """

    sorter_name = "simple"

    handle_multi_segment = True

    _default_params = {
        "apply_preprocessing": False,
        "waveforms": {"ms_before": 1.0, "ms_after": 1.5},
        "filtering": {"freq_min": 300, "freq_max": 8000.0},
        "detection": {"peak_sign": "neg", "detect_threshold": 5.0, "exclude_sweep_ms": 0.4},
        "features": {"n_components": 3},
        "clustering": {
            "method": "hdbscan",
            "min_cluster_size": 25,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
        },
        "job_kwargs": {"n_jobs": -1, "chunk_duration": "1s"},
    }

    @classmethod
    def get_sorter_version(cls):
        return "1.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params["job_kwargs"]
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs.update({"verbose": verbose, "progress_bar": verbose})

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel, cache_preprocessing

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
        from spikeinterface.core.node_pipeline import (
            run_node_pipeline,
            ExtractDenseWaveforms,
            PeakRetriever,
        )

        from sklearn.decomposition import TruncatedSVD

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            recording = bandpass_filter(recording_raw, **params["filtering"], dtype="float32")
            recording = zscore(recording)
            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording = recording_raw
            noise_levels = get_noise_levels(recording, return_scaled=False)

        # detection
        detection_params = params["detection"].copy()
        detection_params["noise_levels"] = noise_levels
        peaks = detect_peaks(recording, method="locally_exclusive", **detection_params, **job_kwargs)

        if verbose:
            print("We found %d peaks in total" % len(peaks))

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]

        # SVD for time compression
        few_peaks = select_peaks(peaks, method="uniform", n_peaks=5000)
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )

        wfs = few_wfs[:, :, 0]
        tsvd = TruncatedSVD(params["features"]["n_components"])
        tsvd.fit(wfs)

        model_folder = sorter_output_folder / "tsvd_model"

        model_folder.mkdir(exist_ok=True)
        with open(model_folder / "pca_model.pkl", "wb") as f:
            pickle.dump(tsvd, f)

        model_params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": float(sampling_frequency),
        }
        with open(model_folder / "params.json", "w") as f:
            json.dump(model_params, f)

        # features

        features_folder = sorter_output_folder / "features"
        node0 = PeakRetriever(recording, peaks)

        node1 = ExtractDenseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
        )

        model_folder_path = sorter_output_folder / "tsvd_model"

        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, model_folder_path=model_folder_path
        )

        pipeline_nodes = [node0, node1, node2]

        output = run_node_pipeline(
            recording,
            pipeline_nodes,
            job_kwargs,
            gather_mode="npy",
            gather_kwargs=dict(exist_ok=True),
            folder=features_folder,
            job_name="extracting features",
            names=["features_tsvd"],
        )

        features_tsvd = np.load(features_folder / "features_tsvd.npy")
        features_flat = features_tsvd.reshape(features_tsvd.shape[0], -1)

        # run hdscan for clustering

        clust_params = params["clustering"].copy()
        clust_method = clust_params.pop("method", "hdbscan")

        if clust_method == "hdbscan":
            import hdbscan
            out = hdbscan.hdbscan(features_flat, **clust_params)
            peak_labels = out[0]
        elif clust_method in ("kmeans"):
            from sklearn.cluster import MiniBatchKMeans
            peak_labels = MiniBatchKMeans(**clust_params).fit_predict(features_flat)
        elif clust_method in ("mean_shift"):
            from sklearn.cluster import MeanShift
            peak_labels = MeanShift().fit_predict(features_flat)
        elif clust_method in ("affinity_propagation"):
            from sklearn.cluster import AffinityPropagation
            peak_labels = AffinityPropagation().fit_predict(features_flat)
        elif clust_method in ("gaussian_mixture"):
            from sklearn.mixture import GaussianMixture
            peak_labels = GaussianMixture(**clust_params).fit_predict(features_flat)
        else:
            raise ValueError(f"simple_sorter : unkown clustering method {clust_method}")

        np.save(features_folder / "peak_labels.npy", peak_labels)

        # keep positive labels
        keep = peak_labels >= 0
        sorting_final = NumpySorting.from_times_labels(
            peaks["sample_index"][keep], peak_labels[keep], sampling_frequency
        )
        sorting_final = sorting_final.save(folder=sorter_output_folder / "sorting")

        return sorting_final
