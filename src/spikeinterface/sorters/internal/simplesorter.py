from .si_based import ComponentsBasedSorter

from spikeinterface.core import load_extractor, BaseRecording, get_noise_levels, extract_waveforms, NumpySorting
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.sortingcomponents.tools import cache_preprocessing
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
        "detection": {"peak_sign": "neg", "detect_threshold": 5.0, "exclude_sweep_ms": 1.5, "radius_um": 150.0},
        "features": {"n_components": 3},
        "clustering": {
            "method": "hdbscan",
            "min_cluster_size": 25,
            "allow_single_cluster": True,
            "core_dist_n_jobs": -1,
            "cluster_selection_method": "leaf",
        },
        # "cache_preprocessing": {"mode": None, "memory_limit": 0.5, "delete_cache": True},
        "job_kwargs": {"n_jobs": -1, "chunk_duration": "1s"},
    }

    _params_description = {
        "apply_preprocessing": "whether to apply the preprocessing steps, default: False",
        "waveforms": "A dictonary containing waveforms params: 'ms_before' (peak of spike) default: 1.0, 'ms_after' (peak of spike) deafult: 1.5",
        "filtering": "A dictionary containing bandpass filter conditions, 'freq_min' default: 300 and 'freq_max' default:8000.0",
        "detection": (
            "A dictionary for specifying the detection conditions of 'peak_sign' (pos or neg) default: 'neg', "
            "'detect_threshold' (snr) default: 5.0, 'exclude_sweep_ms' default: 1.5, 'radius_um' default: 150.0"
        ),
        "features": "A dictionary for the PCA specifying the 'n_components, default: 3",
        "clustering": (
            "A dictionary for specifying the clustering parameters: 'method' (to cluster) default: 'hdbscan', "
            "'min_cluster_size' (min number of spikes per cluster) default: 25, 'allow_single_cluster' default: True, "
            " 'core_dist_n_jobs' (parallelization) default: -1, cluster_selection_method (for hdbscan) default: leaf"
        ),
        "job_kwargs": "Spikeinterface job_kwargs (see job_kwargs documentation) default 'n_jobs': -1, 'chunk_duration': '1s'",
    }

    @classmethod
    def get_sorter_version(cls):
        return "1.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params["job_kwargs"]
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs.update({"progress_bar": verbose})

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel

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

        # recording = cache_preprocessing(recording, **job_kwargs, **params["cache_preprocessing"])

        # detection
        detection_params = params["detection"].copy()
        detection_params["noise_levels"] = noise_levels
        peaks = detect_peaks(recording, method="locally_exclusive", **detection_params, **job_kwargs)

        if verbose:
            print("We found %d peaks in total" % len(peaks))

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]
        nbefore = int(ms_before * sampling_frequency / 1000.0)
        nafter = int(ms_after * sampling_frequency / 1000.0)

        # SVD for time compression

        few_peaks = select_peaks(peaks, recording=recording, method="uniform", n_peaks=5000, margin=(nbefore, nafter))
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

        # folder_to_delete = None

        # if "mode" in params["cache_preprocessing"]:
        #     cache_mode = params["cache_preprocessing"]["mode"]
        # else:
        #     cache_mode = "memory"

        # if "delete_cache" in params["cache_preprocessing"]:
        #     delete_cache = params["cache_preprocessing"]
        # else:
        #     delete_cache = True

        # if cache_mode in ["folder", "zarr"] and delete_cache:
        #     folder_to_delete = recording._kwargs["folder_path"]

        # del recording
        # if folder_to_delete is not None:
        #     shutil.rmtree(folder_to_delete)

        # keep positive labels
        keep = peak_labels >= 0
        sorting_final = NumpySorting.from_times_labels(
            peaks["sample_index"][keep], peak_labels[keep], sampling_frequency
        )
        sorting_final = sorting_final.save(folder=sorter_output_folder / "sorting")

        return sorting_final
