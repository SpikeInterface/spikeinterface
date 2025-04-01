from __future__ import annotations

from pathlib import Path
from packaging import version

from spikeinterface.sorters.basesorter import BaseSorter

from spikeinterface.extractors import HerdingspikesSortingExtractor


class HerdingspikesSorter(BaseSorter):
    """HerdingSpikes Sorter object."""

    sorter_name = "herdingspikes"

    requires_locations = True
    compatible_with_parallel = {"loky": True, "multiprocessing": True, "threading": False}
    _default_params = {
        "chunk_size": None,
        "rescale": True,
        "rescale_value": -1280.0,
        "lowpass": True,
        "common_reference": "median",
        "spike_duration": 1.0,
        "amp_avg_duration": 0.4,
        "threshold": 8.0,
        "min_avg_amp": 1.0,
        "AHP_thr": 0.0,
        "neighbor_radius": 90.0,
        "inner_radius": 70.0,
        "peak_jitter": 0.25,
        "rise_duration": 0.26,
        "decay_filtering": False,
        "decay_ratio": 1.0,
        "localize": True,
        "save_shape": True,
        "out_file": "HS2_detected",
        "left_cutout_time": 0.3,
        "right_cutout_time": 1.8,
        "verbose": True,
        "clustering_bandwidth": 4.0,
        "clustering_alpha": 4.5,
        "clustering_n_jobs": -1,
        "clustering_bin_seeding": True,
        "clustering_min_bin_freq": 4,
        "clustering_subset": None,
        "pca_ncomponents": 2,
        "pca_whiten": True,
    }

    _params_description = {
        "localize": "Perform spike localization. (`bool`, `True`)",
        "save_shape": "Save spike shape. (`bool`, `True`)",
        "out_file": "Path and filename to store detection and clustering results. (`str`, `HS2_detected`)",
        "verbose": "Print progress information. (`bool`, `True`)",
        "chunk_size": " Number of samples per chunk during detection. If `None`, a suitable value will be estimated. (`int`, `None`)",
        "lowpass": "Enable internal low-pass filtering (simple two-step average). (`bool`, `True`)",
        "common_reference": "Method for common reference filtering, can be `average` or `median` (`str`, `median`)",
        "rescale": "Automatically re-scale the data.  (`bool`, `True`)",
        "rescale_value": "Factor by which data is re-scaled. (`float`, `-1280.0`)",
        "threshold": "Spike detection threshold. (`float`, `8.0`)",
        "spike_duration": "Maximum duration over which a spike is evaluated (ms). (`float`, `1.0`)",
        "amp_avg_duration": "Maximum duration over which the spike amplitude  is evaluated (ms). (`float`, `0.4`)",
        "min_avg_amp": "Minimum integrated spike amplitude for a true spike. (`float`, `1.0`)",
        "AHP_thr": "Minimum value of the spike repolarisation for a true spike. (`float`, `0.0`)",
        "neighbor_radius": "Radius of area around probe channel for neighbor classification (microns). (`float`, `90.0`)",
        "inner_radius": "Radius of area around probe channel for spike localisation (microns). (`float`, `70.0`)",
        "peak_jitter": "Maximum peak misalignment for synchronous spike (ms). (`float`, `0.25`)",
        "rise_duration": "Maximum spike rise time, in milliseconds. (`float`, `0.26`)",
        "decay_filtering": "Exclude duplicate spikes based on spatial decay pattern, experimental. (`bool`,`False`)",
        "decay_ratio": "Spatial decay rate for `decay_filtering`. (`float`,`1.0`)",
        "left_cutout_time": "Length of cutout before peak (ms). (`float`, `0.3`)",
        "right_cutout_time": "Length of cutout after peak (ms). (`float`, `1.8`)",
        "pca_ncomponents": "Number of principal components to use when clustering. (`int`, `2`)",
        "pca_whiten": "If `True`, whiten data for PCA. (`bool`, `True`)",
        "clustering_bandwidth": "Meanshift bandwidth, average spatial extent of spike clusters (microns). (`float`, `4.0`)",
        "clustering_alpha": "Scalar for the waveform PC features when clustering. (`float`, `4.5`)",
        "clustering_n_jobs": "Number of cores to use for clustering, use `-1` for all available cores. (`int`, `-1`)",
        "clustering_bin_seeding": "Enable clustering bin seeding. (`bool`, `True`)",
        "clustering_min_bin_freq": "Minimum spikes per bin for bin seeding. (`int`, `4`)",
        "clustering_subset": "Number of spikes used to build clusters. All by default. (`int`, `None`)",
    }

    sorter_description = """Herding Spikes is a density-based spike sorter designed for large-scale high-density recordings.
    It uses both PCA features and an estimate of the spike location to cluster different units.
    For more information see https://www.sciencedirect.com/science/article/pii/S221112471730236X"""

    installation_mesg = """\nTo use HerdingSpikes run:\n
       >>> pip install herdingspikes
    More information on HerdingSpikes at:
      * https://github.com/mhhennig/hs2
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        import importlib.util

        spec = importlib.util.find_spec("herdingspikes")
        if spec is None:
            HAVE_HS = False
        else:
            HAVE_HS = True
        return HAVE_HS

    @classmethod
    def get_sorter_version(cls):
        import herdingspikes as hs

        return hs.__version__

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        # nothing to copy inside the folder : Herdingspikes uses spikeinterface natively
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import herdingspikes as hs

        hs_version = version.parse(hs.__version__)

        if hs_version >= version.parse("0.4.1"):
            lightning_api = True
        else:
            lightning_api = False

        assert (
            lightning_api
        ), "HerdingSpikes version <0.4.1 is no longer supported. To upgrade, run:\n>>> pip install --upgrade herdingspikes"

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        sorted_file = str(sorter_output_folder / "HS2_sorted.hdf5")
        params["out_file"] = str(sorter_output_folder / "HS2_detected")
        p = params
        p.update({"verbose": verbose})

        det = hs.HSDetectionLightning(recording, p)
        det.DetectFromRaw()
        C = hs.HSClustering(det)
        C.ShapePCA()
        C.CombinedClustering(
            alpha=p["clustering_alpha"],
            cluster_subset=p["clustering_subset"],
            bandwidth=p["clustering_bandwidth"],
            bin_seeding=p["clustering_bin_seeding"],
            min_bin_freq=p["clustering_min_bin_freq"],
            n_jobs=p["clustering_n_jobs"],
        )

        if verbose:
            print("Saving to", sorted_file)
        C.SaveHDF5(sorted_file, sampling=recording.get_sampling_frequency())

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return HerdingspikesSortingExtractor(
            file_path=Path(sorter_output_folder) / "HS2_sorted.hdf5", load_unit_info=True
        )
