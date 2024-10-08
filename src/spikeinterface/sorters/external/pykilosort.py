from __future__ import annotations

from pathlib import Path
import numpy as np
import warnings

from spikeinterface.core import load_extractor
from spikeinterface.extractors import KiloSortSortingExtractor
from spikeinterface.core import write_binary_recording
import json
from ..basesorter import BaseSorter, get_job_kwargs


class PyKilosortSorter(BaseSorter):
    """Pykilosort Sorter object."""

    sorter_name = "pykilosort"
    requires_locations = False
    gpu_capability = "nvidia-required"
    requires_binary_data = True
    compatible_with_parallel = {"loky": True, "multiprocessing": False, "threading": False}

    _default_params = {
        "low_memory": False,
        "seed": 42,
        "preprocessing_function": "kilosort2",
        "save_drift_spike_detections": False,
        "perform_drift_registration": False,
        "do_whitening": True,
        "save_temp_files": True,
        "fshigh": 300.0,
        "fslow": None,
        "minfr_goodchannels": 0.1,
        "genericSpkTh": 8.0,
        "nblocks": 5,
        "sig_datashift": 20.0,
        "stable_mode": True,
        "deterministic_mode": True,
        "datashift": None,
        "Th": [10, 4],
        "ThPre": 8,
        "lam": 10,
        "minFR": 1.0 / 50,
        "momentum": [20, 400],
        "sigmaMask": 30,
        "spkTh": -6,
        "reorder": 1,
        "nSkipCov": 25,
        "ntbuff": 64,
        "whiteningRange": 32,
        "scaleproc": 200,
        "nPCs": 3,
        "nt0": 61,
        "nup": 10,
        "sig": 1,
        "gain": 1,
        "templateScaling": 20.0,
        "loc_range": [5, 4],
        "long_range": [30, 6],
        "keep_good_only": False,
    }

    _params_description = {
        "low_memory": "low memory setting for running chronic recordings",
        "seed": "seed for deterministic output",
        "preprocessing_function": 'pre-processing function used choices are "kilosort2" or "destriping"',
        "save_drift_spike_detections": "save detected spikes in drift correction",
        "perform_drift_registration": "Estimate electrode drift and apply registration",
        "do_whitening": "whether or not to whiten data, if disabled channels are individually z-scored",
        "fs": "sample rate",
        "probe": "data type of raw data",
        "data_dtype": "data type of raw data",
        "save_temp_files": "keep temporary files created while running",
        "fshigh": "high pass filter frequency",
        "fslow": "low pass filter frequency",
        "minfr_goodchannels": "minimum firing rate on a 'good' channel (0 to skip)",
        "genericSpkTh": "threshold for crossings with generic templates",
        "nblocks": "number of blocks used to segment the probe when tracking drift, 0 == don't track, 1 == rigid, > 1 == non-rigid",
        "output_filename": "optionally save registered data to a new binary file",
        "overwrite": "overwrite proc file with shifted data",
        "sig_datashift": "sigma for the Gaussian process smoothing",
        "stable_mode": "make output more stable",
        "deterministic_mode": "make output deterministic by sorting spikes before applying kernels",
        "datashift": "parameters for 'datashift' drift correction. not required",
        "Th": "threshold on projections (like in Kilosort1, can be different for last pass like [10 4])",
        "ThPre": "threshold crossings for pre-clustering (in PCA projection space)",
        "lam": "how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)",
        "minFR": " minimum spike rate (Hz), if a cluster falls below this for too long it gets removed",
        "momentum": "number of samples to average over (annealed from first to second value)",
        "sigmaMask": "spatial constant in um for computing residual variance of spike",
        "spkTh": "spike threshold in standard deviations",
        "reorder": "whether to reorder batches for drift correction.",
        "nSkipCov": "compute whitening matrix from every nth batch",
        "ntbuff": "samples of symmetrical buffer for whitening and spike detection; Must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).",
        "whiteningRange": "number of channels to use for whitening each channel",
        "scaleproc": "int16 scaling of whitened data",
        "nPCs": "how many PCs to project the spikes into",
        "nt0": None,
        "nup": None,
        "sig": None,
        "gain": None,
        "templateScaling": None,
        "loc_range": None,
        "long_range": None,
        "keep_good_only": "If True only 'good' units are returned",
    }

    sorter_description = """pykilosort is a port of kilosort to python"""

    installation_mesg = """\nTo use pykilosort:\n
       >>> pip install cupy
        >>> git clone https://github.com/MouseLand/pykilosort
        >>> cd pykilosort
        >>>python setup.py install
    More info at https://github.com/MouseLand/pykilosort#installation
    """

    #
    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        try:
            import pykilosort

            HAVE_PYKILOSORT = True
        except ImportError:
            HAVE_PYKILOSORT = False

        return HAVE_PYKILOSORT

    @classmethod
    def get_sorter_version(cls):
        import pykilosort

        return pykilosort.__version__

    @classmethod
    def _check_params(cls, recording, sorter_output_folder, params):
        return params

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        if not recording.binary_compatible_with(time_axis=0, file_paths_lenght=1):
            # local copy needed
            write_binary_recording(
                recording,
                file_paths=sorter_output_folder / "recording.dat",
                **get_job_kwargs(params, verbose),
            )

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        from pykilosort import Bunch, run

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        if not recording.binary_compatible_with(time_axis=0, file_paths_lenght=1):
            # saved by setup recording
            dat_path = sorter_output_folder / "recording.dat"
        else:
            # no copy
            d = recording.get_binary_description()
            dat_path = d["file_paths"][0]

        num_chans = recording.get_num_channels()
        locations = recording.get_channel_locations()
        params["n_channels"] = num_chans

        # ks_probe is not probeinterface Probe at all
        ks_probe = Bunch()
        # handle different versions
        # Mouseland - develop version
        ks_probe.n_channels = num_chans
        ks_probe.n_channels_tot = num_chans
        ks_probe.channel_map = np.arange(num_chans)
        ks_probe.channel_groups = np.ones(num_chans)
        ks_probe.xcoords = locations[:, 0]
        ks_probe.ycoords = locations[:, 1]
        # IBL version
        ks_probe.Nchans = num_chans
        ks_probe.NchanTOT = num_chans
        ks_probe.chanMap = np.arange(num_chans)
        ks_probe.kcoords = np.ones(num_chans)
        ks_probe.xc = locations[:, 0]
        ks_probe.yc = locations[:, 1]
        ks_probe.shank = None
        ks_probe.channel_labels = np.zeros(num_chans, dtype=int)

        if recording.get_channel_gains() is not None:
            gains = recording.get_channel_gains()
            if len(np.unique(gains)) == 1:
                ks_probe.sample2volt = gains[0] * 1e-6
            else:
                warnings.warn("Multiple gains detected for different channels. Median gain will be used")
                ks_probe.sample2volt = np.median(gains) * 1e-6
        else:
            ks_probe.sample2volt = 1e-6

        run(
            dat_path,
            dir_path=sorter_output_folder,
            probe=ks_probe,
            data_dtype=str(recording.get_dtype()),
            fs=recording.get_sampling_frequency(),
            **params,
        )

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        if (sorter_output_folder.parent / "spikeinterface_params.json").is_file():
            params_file = sorter_output_folder.parent / "spikeinterface_params.json"
        else:
            # back-compatibility
            params_file = sorter_output_folder / "spikeinterface_params.json"
        with params_file.open("r") as f:
            sorter_params = json.load(f)["sorter_params"]
        keep_good_only = sorter_params.get("keep_good_only", False)
        sorting = KiloSortSortingExtractor(folder_path=sorter_output_folder / "output", keep_good_only=keep_good_only)
        return sorting
