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
        "apply_preprocessing": True,
        "freq_min": 150.0,
        "freq_max": 6000.0,
        "peak_sign": "neg",
        "detect_threshold": 5.0,
        "ms_before": 1.0,
        "ms_after": 1.5,
        "n_svd_components_per_channel": 5,
        "clusterer": "hdbscan",
        "clusterer_kwargs": {},
        "seed": None,
        "job_kwargs": {},
    }

    _params_description = {
        "apply_preprocessing": "Apply internal preprocessing or not",
        "freq_min": "Low frequency for bandpass filter",
        "freq_max": "High frequency for bandpass filter",
        "peak_sign": "Sign of peaks neg/pos/both",
        "detect_threshold": "Treshold for peak detection",
        "n_svd_components_per_channel": "Number of SVD components per channel for clustering",
        "ms_before": "Milliseconds before the spike peak for template matching",
        "ms_after": "Milliseconds after the spike peak for template matching",
        "clusterer": "The clusterer algorithm can be hdbscan | isosplit | kmeans | mean_shift | affinity_propagation | gaussian_mixture",
        "clusterer_kwargs": {},
        "seed": "Seed for random number",
        "job_kwargs": "The famous and fabulous job_kwargs",
    }

    @classmethod
    def get_sorter_version(cls):
        return "2025.12"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params["job_kwargs"]
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs.update({"progress_bar": verbose})
        seed = params["seed"]

        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.waveforms.peak_svd import extract_peaks_svd

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            recording = bandpass_filter(
                recording_raw,
                freq_min=params["freq_min"],
                freq_max=params["freq_max"],
                ftype="bessel",
                filter_order=2,
                dtype="float32",
            )
            recording = zscore(recording)
            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording = recording_raw.astype("float32")
            noise_levels = get_noise_levels(recording, return_in_uV=False)

        # detection
        detection_params = dict(
            peak_sign=params["peak_sign"],
            detect_threshold=params["detect_threshold"],
            exclude_sweep_ms=1.5,
            radius_um=150.0,
            noise_levels=noise_levels,
        )
        peaks = detect_peaks(
            recording, method="locally_exclusive", method_kwargs=detection_params, job_kwargs=job_kwargs
        )

        if verbose:
            print("Simple sorter found %d peaks in total" % len(peaks))

        # features with SVD
        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording,
            peaks,
            ms_before=params["ms_before"],
            ms_after=params["ms_after"],
            n_peaks_fit=5000,
            svd_model=None,
            sparsity_mask=None,
            n_components=params["n_svd_components_per_channel"],
            radius_um=120.0,
            motion_aware=False,
            seed=seed,
            job_kwargs=job_kwargs,
        )
        features_flat = peaks_svd.reshape(peaks_svd.shape[0], -1)

        # run clustering
        clusterer = params["clusterer"]
        clusterer_kwargs = params["clusterer_kwargs"]

        if clusterer == "hdbscan":
            import hdbscan

            out = hdbscan.hdbscan(features_flat, **clusterer_kwargs)
            peak_labels = out[0]

        elif clusterer == "isosplit":
            from spikeinterface.sortingcomponents.clustering.isosplit_isocut import isosplit

            peak_labels = isosplit(features_flat, **clusterer_kwargs)

        elif clusterer == "hdbscan-gpu":
            from cuml.cluster import HDBSCAN as hdbscan

            model = hdbscan(**clusterer_kwargs).fit(features_flat)
            peak_labels = model.labels_.copy()
        elif clusterer in ("kmeans"):
            from sklearn.cluster import MiniBatchKMeans

            peak_labels = MiniBatchKMeans(**clusterer_kwargs).fit_predict(features_flat)
        elif clusterer in ("mean_shift"):
            from sklearn.cluster import MeanShift

            peak_labels = MeanShift().fit_predict(features_flat)
        elif clusterer in ("affinity_propagation"):
            from sklearn.cluster import AffinityPropagation

            peak_labels = AffinityPropagation().fit_predict(features_flat)
        elif clusterer in ("gaussian_mixture"):
            from sklearn.mixture import GaussianMixture

            peak_labels = GaussianMixture(**clusterer_kwargs).fit_predict(features_flat)

        else:
            raise ValueError(f"simple_sorter : unkown clustering method {clusterer}")

        # keep positive labels
        keep = peak_labels >= 0
        sorting_final = NumpySorting.from_samples_and_labels(
            peaks["sample_index"][keep], peak_labels[keep], sampling_frequency
        )
        sorting_final = sorting_final.save(folder=sorter_output_folder / "sorting")

        return sorting_final
