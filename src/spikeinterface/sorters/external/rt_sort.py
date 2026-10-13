import importlib.util
import os
import numpy as np

from ..basesorter import BaseSorter

from spikeinterface.extractors import NumpySorting  # TODO: Create separate sorting extractor for RT-Sort


class RTSortSorter(BaseSorter):
    """RTSort sorter object"""

    sorter_name = "rtsort"
    gpu_capability = "nvidia-required"

    _default_params = {
        "detection_model": "neuropixels",
        "recording_window_ms": None,
        "stringent_thresh": 0.175,
        "loose_thresh": 0.075,
        "inference_scaling_numerator": 15.4,
        "ms_before": 0.5,
        "ms_after": 0.5,
        "pre_median_ms": 50,
        "inner_radius": 50,
        "outer_radius": 100,
        "min_elecs_for_array_noise_n": 100,
        "min_elecs_for_array_noise_f": 0.1,
        "min_elecs_for_seq_noise_n": 50,
        "min_elecs_for_seq_noise_f": 0.05,
        "min_activity_root_cocs": 2,
        "min_activity_hz": 0.05,
        "max_n_components_latency": 4,
        "min_coc_n": 10,
        "min_coc_p": 10,
        "min_extend_comp_p": 50,
        "elec_patience": 6,
        "split_coc_clusters_amps": True,
        "min_amp_dist_p": 0.1,
        "max_n_components_amp": 4,
        "min_loose_elec_prob": 0.03,
        "min_inner_loose_detections": 3,
        "min_loose_detections_n": 4,
        "min_loose_detections_r_spikes": 1 / 3,
        "min_loose_detections_r_sequences": 1 / 3,
        "max_latency_diff_spikes": 2.5,
        "max_latency_diff_sequences": 2.5,
        "clip_latency_diff_factor": 2,
        "max_amp_median_diff_spikes": 0.45,
        "max_amp_median_diff_sequences": 0.45,
        "clip_amp_median_diff_factor": 2,
        "max_root_amp_median_std_spikes": 2.5,
        "max_root_amp_median_std_sequences": 2.5,
        "repeated_detection_overlap_time": 0.2,
        "min_seq_spikes_n": 10,
        "min_seq_spikes_hz": 0.05,
        "relocate_root_min_amp": 0.8,
        "relocate_root_max_latency": -2,
        "device": "cuda",
        "num_processes": None,
        "ignore_warnings": True,
        "debug": False,
    }

    _params_description = {
        "detection_model": "`mea` or `neuropixels` to use the mea- or neuropixels-trained detection models. Or, path to saved detection model (see https://braingeneers.github.io/braindance/docs/RT-sort/usage/training-models) (`str`, `neuropixels`)",
        "recording_window_ms": "A tuple `(start_ms, end_ms)` defining the portion of the recording (in milliseconds) to process. (`tuple`, `None`)",
        "stringent_thresh": "The stringent threshold for spike detection. (`float`, `0.175`)",
        "loose_thresh": "The loose threshold for spike detection. (`float`, `0.075`)",
        "inference_scaling_numerator": "Scaling factor for inference. (`float`, `15.4`)",
        "ms_before": "Time (in milliseconds) to consider before each detected spike for sequence formation. (`float`, `0.5`)",
        "ms_after": "Time (in milliseconds) to consider after each detected spike for sequence formation. (`float`, `0.5`)",
        "pre_median_ms": "Duration (in milliseconds) to compute the median for normalization. (`float`, `50`)",
        "inner_radius": "Inner radius (in micrometers). (`float`, `50`)",
        "outer_radius": "Outer radius (in micrometers). (`float`, `100`)",
        "min_elecs_for_array_noise_n": "Minimum number of electrodes for array-wide noise filtering. (`int`, `100`)",
        "min_elecs_for_array_noise_f": "Minimum fraction of electrodes for array-wide noise filtering. (`float`, `0.1`)",
        "min_elecs_for_seq_noise_n": "Minimum number of electrodes for sequence-wide noise filtering. (`int`, `50`)",
        "min_elecs_for_seq_noise_f": "Minimum fraction of electrodes for sequence-wide noise filtering. (`float`, `0.05`)",
        "min_activity_root_cocs": "Minimum number of stringent spike detections on inner electrodes within the maximum propagation window that cause a stringent spike detection on a root electrode to be counted as a stringent codetection. (`int`, `2`)",
        "min_activity_hz": "Minimum activity rate of root detections (in Hz) for an electrode to be used as a root electrode. (`float`, `0.05`)",
        "max_n_components_latency": "Maximum number of latency components for Gaussian mixture model used for splitting latency distribution. (`int`, `4`)",
        "min_coc_n": "After splitting a cluster of codetections, a cluster is discarded if it does not have at least min_coc_n codetections. (`int`, `10`)",
        "min_coc_p": "After splitting a cluster of codetections, a cluster is discarded if it does not have at least (min_coc_p * the total number of codetections before splitting) codetections. (`int`, `10`)",
        "min_extend_comp_p": "The required percentage of codetections before splitting that is preserved after the split in order for the inner electrodes of the current splitting electrode to be added to the total list of electrodes used to further split the cluster. (`int`, `50`)",
        "elec_patience": "Number of electrodes considered for splitting that do not lead to a split before terminating the splitting process. (`int`, `6`)",
        "split_coc_clusters_amps": "Whether to split clusters based on amplitude. (`bool`, `True`)",
        "min_amp_dist_p": "The minimum Hartigan's dip test p-value for a distribution to be considered unimodal. (`float`, `0.1`)",
        "max_n_components_amp": "Maximum number of components for Gaussian mixture model used for splitting amplitude distribution. (`int`, `4`)",
        "min_loose_elec_prob": "Minimum average detection score (smaller values are set to 0) in decimal form (ranging from 0 to 1). (`float`, `0.03`)",
        "min_inner_loose_detections": "Minimum inner loose electrode detections for assigning spikes / overlaps for merging. (`int`, `3`)",
        "min_loose_detections_n": "Minimum loose electrode detections for assigning spikes / overlaps for merging. (`int`, `4`)",
        "min_loose_detections_r_spikes": "Minimum ratio of loose electrode detections for assigning spikes. (`float`, `1/3`)",
        "min_loose_detections_r_sequences": "Minimum ratio of loose electrode detections overlaps for merging. (`float`, `1/3`)",
        "max_latency_diff_spikes": "Maximum allowed weighted latency difference for spike assignment. (`float`, `2.5`)",
        "max_latency_diff_sequences": "Maximum allowed weighted latency difference for sequence merging. (`float`, `2.5`)",
        "clip_latency_diff_factor": "Latency clip = clip_latency_diff_factor * max_latency_diff. (`float`, `2`)",
        "max_amp_median_diff_spikes": "Maximum allowed weighted percent amplitude difference for spike assignment. (`float`, `0.45`)",
        "max_amp_median_diff_sequences": "Maximum allowed weighted percent amplitude difference for sequence merging. (`float`, `0.45`)",
        "clip_amp_median_diff_factor": "Amplitude clip = clip_amp_median_diff_factor * max_amp_median_diff. (`float`, `2`)",
        "max_root_amp_median_std_spikes": "Maximum allowed root amplitude standard deviation for spike assignment. (`float`, `2.5`)",
        "max_root_amp_median_std_sequences": "Maximum allowed root amplitude standard deviation for sequence merging. (`float`, `2.5`)",
        "repeated_detection_overlap_time": "Time window (in seconds) for overlapping repeated detections. (`float`, `0.2`)",
        "min_seq_spikes_n": "Minimum number of spikes required for a valid sequence. (`int`, `10`)",
        "min_seq_spikes_hz": "Minimum spike rate for a valid sequence. (`float`, `0.05`)",
        "relocate_root_min_amp": "Minimum amplitude ratio for relocating a root electrode before first merging. (`float`, `0.8`)",
        "relocate_root_max_latency": "Maximum latency for relocating a root electrode before first merging. (`float`, `-2`)",
        "device": "The device for PyTorch operations ('cuda' or 'cpu'). (`str`, `cuda`)",
        "num_processes": "Number of processes to use for parallelization. (`int`, `None`)",
        "ignore_warnings": "Whether to suppress warnings during execution. (`bool`, `True`)",
        "debug": "Whether to enable debugging features such as saving intermediate steps. (`bool`, `False`)",
    }

    sorter_description = """RT-Sort is a real-time spike sorting algorithm that enables the sorted detection of action potentials within 7.5ms±1.5ms (mean±STD) after the waveform trough while the recording remains ongoing.
    It utilizes unique propagation patterns of action potentials along axons detected as high-fidelity sequential activations on adjacent electrodes, together with a convolutional neural network-based spike detection algorithm.
    This implementation in SpikeInterface only implements RT-Sort's offline sorting.
    For more information see https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312438"""

    installation_mesg = f"""\nTo use RTSort run:\n
       >>> pip install git+https://github.com/braingeneers/braindance#egg=braindance[rt-sort]

    Additionally, install PyTorch (https://pytorch.org/get-started/locally/) with any version of CUDA as the compute platform.
    If running on a Linux machine, install Torch-TensorRT (https://pytorch.org/TensorRT/getting_started/installation.html) for faster computations.

    More information on RTSort at: https://github.com/braingeneers/braindance
    """

    handle_multi_segment = False

    @classmethod
    def get_sorter_version(cls):
        import braindance

        return braindance.__version__

    @classmethod
    def is_installed(cls):
        libraries = ["braindance", "torch" if os.name == "nt" else "torch_tensorrt", "diptest", "pynvml", "sklearn"]

        HAVE_RTSORT = True
        for lib in libraries:
            if importlib.util.find_spec(lib) is None:
                HAVE_RTSORT = False
                break

        return HAVE_RTSORT

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        return params

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        # nothing to copy inside the folder : RTSort uses spikeinterface natively
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        from braindance.core.spikesorter.rt_sort import detect_sequences
        from braindance.core.spikedetector.model import ModelSpikeSorter

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        params = params.copy()
        params["recording"] = recording
        rt_sort_inter = sorter_output_folder / "rt_sort_inter"
        params["inter_path"] = rt_sort_inter
        params["verbose"] = verbose

        if params["detection_model"] == "mea":
            params["detection_model"] = ModelSpikeSorter.load_mea()
        elif params["detection_model"] == "neuropixels":
            params["detection_model"] = ModelSpikeSorter.load_neuropixels()
        else:
            params["detection_model"] = ModelSpikeSorter.load(params["detection_model"])

        rt_sort = detect_sequences(**params, delete_inter=False, return_spikes=False)
        np_sorting = rt_sort.sort_offline(
            rt_sort_inter / "scaled_traces.npy",
            verbose=verbose,
            recording_window_ms=params.get("recording_window_ms", None),
            return_spikeinterface_sorter=True,
        )  # type: NumpySorting
        rt_sort.save(sorter_output_folder / "rt_sort.pickle")
        np_sorting.save(folder=sorter_output_folder / "rt_sorting")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return NumpySorting.load_from_folder(sorter_output_folder / "rt_sorting")
