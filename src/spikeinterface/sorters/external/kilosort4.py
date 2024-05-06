from __future__ import annotations

from pathlib import Path
from typing import Union

from ..basesorter import BaseSorter
from .kilosortbase import KilosortBase

PathType = Union[str, Path]


class Kilosort4Sorter(BaseSorter):
    """Kilosort4 Sorter object."""

    sorter_name: str = "kilosort4"
    requires_locations = True
    gpu_capability = "nvidia-optional"

    _default_params = {
        "batch_size": 60000,
        "nblocks": 1,
        "Th_universal": 9,
        "Th_learned": 8,
        "do_CAR": True,
        "invert_sign": False,
        "nt": 61,
        "artifact_threshold": None,
        "nskip": 25,
        "whitening_range": 32,
        "binning_depth": 5,
        "sig_interp": 20,
        "nt0min": None,
        "dmin": None,
        "dminx": 32,
        "min_template_size": 10,
        "template_sizes": 5,
        "nearest_chans": 10,
        "nearest_templates": 100,
        "max_channel_distance": None,
        "templates_from_data": True,
        "n_templates": 6,
        "n_pcs": 6,
        "Th_single_ch": 6,
        "acg_threshold": 0.2,
        "ccg_threshold": 0.25,
        "cluster_downsampling": 20,
        "cluster_pcs": 64,
        "x_centers": None,
        "duplicate_spike_bins": 7,
        "do_correction": True,
        "keep_good_only": False,
        "save_extra_kwargs": False,
        "skip_kilosort_preprocessing": False,
        "scaleproc": None,
        "torch_device": "auto",
    }

    _params_description = {
        "batch_size": "Number of samples included in each batch of data.",
        "nblocks": "Number of non-overlapping blocks for drift correction (additional nblocks-1 blocks are created in the overlaps). Default value: 1.",
        "Th_universal": "Spike detection threshold for universal templates. Th(1) in previous versions of Kilosort. Default value: 9.",
        "Th_learned": "Spike detection threshold for learned templates. Th(2) in previous versions of Kilosort. Default value: 8.",
        "do_CAR": "Whether to perform common average reference. Default value: True.",
        "invert_sign": "Invert the sign of the data. Default value: False.",
        "nt": "Number of samples per waveform. Also size of symmetric padding for filtering. Default value: 61.",
        "artifact_threshold": "If a batch contains absolute values above this number, it will be zeroed out under the assumption that a recording artifact is present. By default, the threshold is infinite (so that no zeroing occurs). Default value: None.",
        "nskip": "Batch stride for computing whitening matrix. Default value: 25.",
        "whitening_range": "Number of nearby channels used to estimate the whitening matrix. Default value: 32.",
        "binning_depth": "For drift correction, vertical bin size in microns used for 2D histogram. Default value: 5.",
        "sig_interp": "For drift correction, sigma for interpolation (spatial standard deviation). Approximate smoothness scale in units of microns. Default value: 20.",
        "nt0min": "Sample index for aligning waveforms, so that their minimum or maximum value happens here. Default of 20. Default value: None.",
        "dmin": "Vertical spacing of template centers used for spike detection, in microns. Determined automatically by default. Default value: None.",
        "dminx": "Horizontal spacing of template centers used for spike detection, in microns. Default value: 32.",
        "min_template_size": "Standard deviation of the smallest, spatial envelope Gaussian used for universal templates. Default value: 10.",
        "template_sizes": "Number of sizes for universal spike templates (multiples of the min_template_size). Default value: 5.",
        "nearest_chans": "Number of nearest channels to consider when finding local maxima during spike detection. Default value: 10.",
        "nearest_templates": "Number of nearest spike template locations to consider when finding local maxima during spike detection. Default value: 100.",
        "max_channel_distance": "Templates farther away than this from their nearest channel will not be used. Also limits distance between compared channels during clustering. Default value: None.",
        "templates_from_data": "Indicates whether spike shapes used in universal templates should be estimated from the data or loaded from the predefined templates. Default value: True.",
        "n_templates": "Number of single-channel templates to use for the universal templates (only used if templates_from_data is True). Default value: 6.",
        "n_pcs": "Number of single-channel PCs to use for extracting spike features (only used if templates_from_data is True). Default value: 6.",
        "Th_single_ch": "For single channel threshold crossings to compute universal- templates. In units of whitened data standard deviations. Default value: 6.",
        "acg_threshold": 'Fraction of refractory period violations that are allowed in the ACG compared to baseline; used to assign "good" units. Default value: 0.2.',
        "ccg_threshold": "Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges. Default value: 0.25.",
        "cluster_downsampling": "Inverse fraction of nodes used as landmarks during clustering (can be 1, but that slows down the optimization). Default value: 20.",
        "cluster_pcs": "Maximum number of spatiotemporal PC features used for clustering. Default value: 64.",
        "x_centers": "Number of x-positions to use when determining center points for template groupings. If None, this will be determined automatically by finding peaks in channel density. For 2D array type probes, we recommend specifying this so that centers are placed every few hundred microns.",
        "duplicate_spike_bins": "Number of bins for which subsequent spikes from the same cluster are assumed to be artifacts. A value of 0 disables this step. Default value: 7.",
        "keep_good_only": "If True only 'good' units are returned",
        "do_correction": "If True, drift correction is performed",
        "save_extra_kwargs": "If True, additional kwargs are saved to the output",
        "skip_kilosort_preprocessing": "Can optionally skip the internal kilosort preprocessing",
        "scaleproc": "int16 scaling of whitened data, if None set to 200.",
        "torch_device": "Select the torch device auto/cuda/cpu",
    }

    sorter_description = """Kilosort4 is a Python package for spike sorting on GPUs with template matching.
    The software uses new graph-based approaches to clustering that improve performance compared to previous versions.
    For detailed comparisons to past versions of Kilosort and to other spike-sorting methods, please see the pre-print
    at https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1
    For more information see https://github.com/MouseLand/Kilosort"""

    installation_mesg = """\nTo use Kilosort4 run:\n
        >>> pip install kilosort==4.0

    More information on Kilosort4 at:
        https://github.com/MouseLand/Kilosort
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        try:
            import kilosort as ks
            import torch

            HAVE_KS = True
        except ImportError:
            HAVE_KS = False
        return HAVE_KS

    @classmethod
    def get_sorter_version(cls):
        import kilosort as ks

        return ks.__version__

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        from probeinterface import write_prb

        pg = recording.get_probegroup()
        probe_filename = sorter_output_folder / "probe.prb"
        write_prb(probe_filename, pg)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        from kilosort.run_kilosort import (
            set_files,
            initialize_ops,
            compute_preprocessing,
            compute_drift_correction,
            detect_spikes,
            cluster_spikes,
            save_sorting,
            get_run_parameters,
        )
        from kilosort.io import load_probe, RecordingExtractorAsArray, BinaryFiltered
        from kilosort.parameters import DEFAULT_SETTINGS

        import time
        import torch
        import numpy as np

        sorter_output_folder = sorter_output_folder.absolute()

        probe_filename = sorter_output_folder / "probe.prb"

        torch_device = params["torch_device"]
        if torch_device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(torch_device)

        # load probe
        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        probe = load_probe(probe_filename)
        probe_name = ""
        filename = ""

        # this internally concatenates the recording
        file_object = RecordingExtractorAsArray(recording)

        do_CAR = params["do_CAR"]
        invert_sign = params["invert_sign"]
        save_extra_vars = params["save_extra_kwargs"]
        progress_bar = None
        settings_ks = {k: v for k, v in params.items() if k in DEFAULT_SETTINGS}
        settings_ks["n_chan_bin"] = recording.get_num_channels()
        settings_ks["fs"] = recording.sampling_frequency
        if not do_CAR:
            print("Skipping common average reference.")

        tic0 = time.time()

        settings = {**DEFAULT_SETTINGS, **settings_ks}

        if settings["nt0min"] is None:
            settings["nt0min"] = int(20 * settings["nt"] / 61)
        if settings["artifact_threshold"] is None:
            settings["artifact_threshold"] = np.inf

        # NOTE: Also modifies settings in-place
        data_dir = ""
        results_dir = sorter_output_folder
        filename, data_dir, results_dir, probe = set_files(settings, filename, probe, probe_name, data_dir, results_dir)
        ops = initialize_ops(settings, probe, recording.get_dtype(), do_CAR, invert_sign, device)

        n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, _, _, tmin, tmax, artifact = (
            get_run_parameters(ops)
        )
        # Set preprocessing and drift correction parameters
        if not params["skip_kilosort_preprocessing"]:
            ops = compute_preprocessing(ops, device, tic0=tic0, file_object=file_object)
        else:
            print("Skipping kilosort preprocessing.")
            bfile = BinaryFiltered(
                ops["filename"],
                n_chan_bin,
                fs,
                NT,
                nt,
                twav_min,
                chan_map,
                hp_filter=None,
                device=device,
                do_CAR=do_CAR,
                invert_sign=invert,
                dtype=dtype,
                tmin=tmin,
                tmax=tmax,
                artifact_threshold=artifact,
                file_object=file_object,
            )
            ops["preprocessing"] = dict(hp_filter=None, whiten_mat=None)
            ops["Wrot"] = torch.as_tensor(np.eye(recording.get_num_channels()))
            ops["Nbatches"] = bfile.n_batches

        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1)
        # if not params["skip_kilosort_preprocessing"]:
        if not params["do_correction"]:
            print("Skipping drift correction.")
            ops["nblocks"] = 0

        # this function applies both preprocessing and drift correction
        ops, bfile, st0 = compute_drift_correction(
            ops, device, tic0=tic0, progress_bar=progress_bar, file_object=file_object
        )

        # Sort spikes and save results
        st, tF, _, _ = detect_spikes(ops, device, bfile, tic0=tic0, progress_bar=progress_bar)
        clu, Wall = cluster_spikes(st, tF, ops, device, bfile, tic0=tic0, progress_bar=progress_bar)
        if params["skip_kilosort_preprocessing"]:
            ops["preprocessing"] = dict(
                hp_filter=torch.as_tensor(np.zeros(1)), whiten_mat=torch.as_tensor(np.eye(recording.get_num_channels()))
            )

        _ = save_sorting(ops, results_dir, st, clu, tF, Wall, bfile.imin, tic0, save_extra_vars=save_extra_vars)

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return KilosortBase._get_result_from_folder(sorter_output_folder)
