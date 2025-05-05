from __future__ import annotations

from pathlib import Path
from packaging.version import parse

import shutil
from warnings import warn
import importlib.util
from importlib.metadata import version

import numpy as np

from spikeinterface.preprocessing import bandpass_filter, whiten
from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.sorters.basesorter import BaseSorter, get_job_kwargs
from spikeinterface.extractors import NpzSortingExtractor


class Mountainsort5Sorter(BaseSorter):
    """Mountainsort5 Sorter object."""

    sorter_name = "mountainsort5"
    requires_locations = False
    compatible_with_parallel = {"loky": False, "multiprocessing": False, "threading": False}
    requires_binary_data = True

    _default_params = {
        "scheme": "2",  # '1', '2', '3'
        "detect_threshold": 5.5,  # this is the recommended detection threshold
        "detect_sign": -1,
        "detect_time_radius_msec": 0.5,
        "snippet_T1": 20,
        "snippet_T2": 20,
        "npca_per_channel": 3,
        "npca_per_subdivision": 10,
        "snippet_mask_radius": 250,
        "scheme1_detect_channel_radius": 150,
        "scheme2_phase1_detect_channel_radius": 200,
        "scheme2_detect_channel_radius": 50,
        "scheme2_max_num_snippets_per_training_batch": 200,
        "scheme2_training_duration_sec": 60 * 5,
        "scheme2_training_recording_sampling_mode": "uniform",
        "scheme3_block_duration_sec": 60 * 30,
        "freq_min": 300,
        "freq_max": 6000,
        "filter": True,
        "whiten": True,  # Important to do whitening
        "delete_temporary_recording": True,
    }

    _params_description = {
        "scheme": "Which sorting scheme to use: '1, '2', or '3'",
        "detect_threshold": "Detection threshold - recommend to use the default",
        "detect_sign": "Use -1 for detecting negative peaks, 1 for positive, 0 for both",
        "detect_time_radius_msec": "Determines the minimum allowable time interval between detected spikes in the same spatial region",
        "snippet_T1": "Number of samples before the peak to include in the snippet",
        "snippet_T2": "Number of samples after the peak to include in the snippet",
        "npca_per_channel": "Number of PCA features per channel in the initial dimension reduction step",
        "npca_per_subdivision": "Number of PCA features to compute at each stage of clustering in the isosplit6 subdivision method",
        "snippet_mask_radius": "Radius of the mask to apply to the extracted snippets",
        "scheme1_detect_channel_radius": "Channel radius for excluding events that are too close in time in scheme 1",
        "scheme2_phase1_detect_channel_radius": "Channel radius for excluding events that are too close in time during phase 1 of scheme 2",
        "scheme2_detect_channel_radius": "Channel radius for excluding events that are too close in time during phase 2 of scheme 2",
        "scheme2_max_num_snippets_per_training_batch": "Maximum number of snippets to use in each batch for training during phase 2 of scheme 2",
        "scheme2_training_duration_sec": "Duration of training data to use in scheme 2",
        "scheme2_training_recording_sampling_mode": "initial or uniform",
        "scheme3_block_duration_sec": "Duration of each block in scheme 3",
        "freq_min": "High-pass filter cutoff frequency",
        "freq_max": "Low-pass filter cutoff frequency",
        "filter": "Enable or disable filter",
        "whiten": "Enable or disable whitening",
        "delete_temporary_recording": "If True, the temporary recording file is deleted after sorting (this may fail on Windows requiring the end-user to delete the file themselves later)",
    }

    sorter_description = "MountainSort5 uses Isosplit clustering. It is an updated version of MountainSort4. See https://doi.org/10.1016/j.neuron.2017.08.030"

    installation_mesg = """\nTo use Mountainsort5 run:\n
       >>> pip install mountainsort5

    More information on mountainsort5 at:
      * https://github.com/flatironinstitute/mountainsort5
    """

    @classmethod
    def is_installed(cls):
        ms5_spec = importlib.util.find_spec("mountainsort5")
        if ms5_spec is not None:
            HAVE_MS5 = True
        else:
            HAVE_MS5 = False

        if HAVE_MS5:
            ms5_version = version("mountainsort5")
            vv = parse(ms5_version)
            if vv < parse("0.3"):
                warn(
                    f"WARNING: This version of SpikeInterface expects Mountainsort5 version 0.3.x or newer. "
                    f"You have version {ms5_version}"
                )
                HAVE_MS5 = False
        return HAVE_MS5

    @staticmethod
    def get_sorter_version():
        import mountainsort5

        if hasattr(mountainsort5, "__version__"):
            return mountainsort5.__version__
        return "unknown"

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params["filter"]

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import mountainsort5 as ms5

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        if recording is None:
            raise Exception("Unable to load recording from folder.")
        if not isinstance(recording, BaseRecording):
            raise TypeError(
                f"Unexpected: recording extracted from folder is not a BaseRecording, but is of type: {type(recording)}"
            )

        # alias to params
        p = params

        # Bandpass filter
        if p["filter"] and p["freq_min"] is not None and p["freq_max"] is not None:
            if verbose:
                print("filtering")
            # important to use dtype=float here
            recording = bandpass_filter(
                recording=recording, freq_min=p["freq_min"], freq_max=p["freq_max"], dtype=np.float32
            )

        # Whiten
        if p["whiten"]:
            if verbose:
                print("whitening")
            recording = whiten(recording=recording, dtype="float32")
        else:
            warn("Not whitening (MountainSort5 expects whitened data)")

        scheme1_sorting_parameters = ms5.Scheme1SortingParameters(
            detect_threshold=p["detect_threshold"],
            detect_channel_radius=p["scheme1_detect_channel_radius"],
            detect_time_radius_msec=p["detect_time_radius_msec"],
            detect_sign=p["detect_sign"],
            snippet_T1=p["snippet_T1"],
            snippet_T2=p["snippet_T2"],
            snippet_mask_radius=p["snippet_mask_radius"],
            npca_per_channel=p["npca_per_channel"],
            npca_per_subdivision=p["npca_per_subdivision"],
        )

        scheme2_sorting_parameters = ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=p["scheme2_phase1_detect_channel_radius"],
            detect_channel_radius=p["scheme2_detect_channel_radius"],
            phase1_detect_threshold=p["detect_threshold"],
            phase1_detect_time_radius_msec=p["detect_time_radius_msec"],
            detect_time_radius_msec=p["detect_time_radius_msec"],
            phase1_npca_per_channel=p["npca_per_channel"],
            phase1_npca_per_subdivision=p["npca_per_subdivision"],
            detect_sign=p["detect_sign"],
            detect_threshold=p["detect_threshold"],
            snippet_T1=p["snippet_T1"],
            snippet_T2=p["snippet_T2"],
            snippet_mask_radius=p["snippet_mask_radius"],
            max_num_snippets_per_training_batch=p["scheme2_max_num_snippets_per_training_batch"],
            classifier_npca=None,
            training_duration_sec=p["scheme2_training_duration_sec"],
            training_recording_sampling_mode=p["scheme2_training_recording_sampling_mode"],
        )

        scheme3_sorting_parameters = ms5.Scheme3SortingParameters(
            block_sorting_parameters=scheme2_sorting_parameters, block_duration_sec=p["scheme3_block_duration_sec"]
        )

        if not recording.is_binary_compatible():
            recording_cached = recording.save(folder=sorter_output_folder / "recording", **get_job_kwargs(p, verbose))
        else:
            recording_cached = recording

        if p["scheme"] == "1":
            sorting = ms5.sorting_scheme1(recording=recording_cached, sorting_parameters=scheme1_sorting_parameters)
        elif p["scheme"] == "2":
            sorting = ms5.sorting_scheme2(recording=recording_cached, sorting_parameters=scheme2_sorting_parameters)
        elif p["scheme"] == "3":
            sorting = ms5.sorting_scheme3(recording=recording_cached, sorting_parameters=scheme3_sorting_parameters)
        else:
            raise ValueError(f"Invalid scheme: {p['scheme']} given. scheme must be one of '1', '2' or '3'")

        if p["delete_temporary_recording"]:
            if not recording.is_binary_compatible():
                del recording_cached
                shutil.rmtree(sorter_output_folder / "recording", ignore_errors=True)
                if Path(sorter_output_folder / "recording").is_dir():
                    warn("cleanup failed, please remove file yourself if desired")

        NpzSortingExtractor.write_sorting(sorting, str(sorter_output_folder / "firings.npz"))

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        result_fname = sorter_output_folder / "firings.npz"
        sorting = NpzSortingExtractor(result_fname)
        return sorting
