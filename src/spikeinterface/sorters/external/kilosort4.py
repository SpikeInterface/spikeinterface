from __future__ import annotations

from pathlib import Path
from typing import Union
from packaging import version


from ...core import write_binary_recording
from ..basesorter import BaseSorter, get_job_kwargs
from .kilosortbase import KilosortBase
from ..basesorter import get_job_kwargs
from importlib.metadata import version as importlib_version

PathType = Union[str, Path]


class Kilosort4Sorter(BaseSorter):
    """Kilosort4 Sorter object."""

    sorter_name: str = "kilosort4"
    requires_locations = True
    gpu_capability = "nvidia-optional"
    requires_binary_data = False

    _default_params = {
        "batch_size": 60000,
        "nblocks": 1,
        "Th_universal": 9,
        "Th_learned": 8,
        "do_CAR": True,
        "invert_sign": False,
        "nt": 61,
        "shift": None,
        "scale": None,
        "artifact_threshold": None,
        "nskip": 25,
        "whitening_range": 32,
        "highpass_cutoff": 300,
        "binning_depth": 5,
        "sig_interp": 20,
        "drift_smoothing": [0.5, 0.5, 0.5],
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
        "duplicate_spike_ms": 0.25,
        "scaleproc": None,
        "save_preprocessed_copy": False,
        "torch_device": "auto",
        "bad_channels": None,
        "clear_cache": False,
        "save_extra_vars": False,
        "do_correction": True,
        "keep_good_only": False,
        "skip_kilosort_preprocessing": False,
        "use_binary_file": None,
        "delete_recording_dat": True,
    }

    _params_description = {
        "batch_size": "Number of samples included in each batch of data.",
        "nblocks": "Number of non-overlapping blocks for drift correction (additional nblocks-1 blocks are created in the overlaps). Default value: 1.",
        "Th_universal": "Spike detection threshold for universal templates. Th(1) in previous versions of Kilosort. Default value: 9.",
        "Th_learned": "Spike detection threshold for learned templates. Th(2) in previous versions of Kilosort. Default value: 8.",
        "do_CAR": "Whether to perform common average reference. Default value: True.",
        "invert_sign": "Invert the sign of the data. Default value: False.",
        "nt": "Number of samples per waveform. Also size of symmetric padding for filtering. Default value: 61.",
        "shift": "Scalar shift to apply to data before all other operations. Default None.",
        "scale": "Scaling factor to apply to data before all other operations. Default None.",
        "artifact_threshold": "If a batch contains absolute values above this number, it will be zeroed out under the assumption that a recording artifact is present. By default, the threshold is infinite (so that no zeroing occurs). Default value: None.",
        "nskip": "Batch stride for computing whitening matrix. Default value: 25.",
        "whitening_range": "Number of nearby channels used to estimate the whitening matrix. Default value: 32.",
        "highpass_cutoff": "High-pass filter cutoff frequency in Hz. Default value: 300.",
        "binning_depth": "For drift correction, vertical bin size in microns used for 2D histogram. Default value: 5.",
        "sig_interp": "For drift correction, sigma for interpolation (spatial standard deviation). Approximate smoothness scale in units of microns. Default value: 20.",
        "drift_smoothing": "Amount of gaussian smoothing to apply to the spatiotemporal drift estimation, for x,y,time axes in units of registration blocks (for x,y axes) and batch size (for time axis). The x,y smoothing has no effect for `nblocks = 1`.",
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
        "save_extra_vars": "If True, additional kwargs are saved to the output",
        "scaleproc": "int16 scaling of whitened data, if None set to 200.",
        "save_preprocessed_copy": "Save a pre-processed copy of the data (including drift correction) to temp_wh.dat in the results directory and format Phy output to use that copy of the data",
        "torch_device": "Select the torch device auto/cuda/cpu",
        "bad_channels": "A list of channel indices (rows in the binary file) that should not be included in sorting. Listing channels here is equivalent to excluding them from the probe dictionary.",
        "clear_cache": "If True, force pytorch to free up memory reserved for its cache in between memory-intensive operations. Note that setting `clear_cache=True` is NOT recommended unless you encounter GPU out-of-memory errors, since this can result in slower sorting.",
        "do_correction": "If True, drift correction is performed. Default is True. (spikeinterface parameter)",
        "skip_kilosort_preprocessing": "Can optionally skip the internal kilosort preprocessing. (spikeinterface parameter)",
        "keep_good_only": "If True, only the units labeled as 'good' by Kilosort are returned in the output. (spikeinterface parameter)",
        "use_binary_file": "If True then Kilosort is run using a binary file. In this case, if the input recording is not binary compatible, it is written to a binary file in the output folder. "
        "If False then Kilosort is run on the recording object directly using the RecordingExtractorAsArray object. If None, then if the recording is binary compatible, the sorter will use the binary file, otherwise the RecordingExtractorAsArray. "
        "Default is None. (spikeinterface parameter)",
        "delete_recording_dat": "If True, if a temporary binary file is created, it is deleted after the sorting is done. Default is True. (spikeinterface parameter)",
    }

    sorter_description = """Kilosort4 is a Python package for spike sorting on GPUs with template matching.
    The software uses new graph-based approaches to clustering that improve performance compared to previous versions.
    For detailed comparisons to past versions of Kilosort and to other spike-sorting methods, please see the pre-print
    at https://www.biorxiv.org/content/10.1101/2023.01.07.523036v1
    For more information see https://github.com/MouseLand/Kilosort"""

    installation_mesg = """\nTo use Kilosort4 run:\n
        >>> pip install kilosort --upgrade

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
        """kilosort.__version__ <4.0.10 is always '4'"""
        return importlib_version("kilosort")

    @classmethod
    def initialize_folder(cls, recording, output_folder, verbose, remove_existing_folder):
        if not cls.is_installed():
            raise Exception(
                f"The sorter {cls.sorter_name} is not installed. Please install it with:\n{cls.installation_mesg}"
            )
        cls.check_sorter_version()
        return super(Kilosort4Sorter, cls).initialize_folder(recording, output_folder, verbose, remove_existing_folder)

    @classmethod
    def check_sorter_version(cls):
        kilosort_version = version.parse(cls.get_sorter_version())
        if kilosort_version < version.parse("4.0.16"):
            raise Exception(
                f"""SpikeInterface only supports kilosort versions 4.0.16 and above. You are running version {kilosort_version}. To install the latest version, run:
                        >>> pip install kilosort --upgrade
                """
            )

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        from probeinterface import write_prb

        pg = recording.get_probegroup()
        probe_filename = sorter_output_folder / "probe.prb"
        write_prb(probe_filename, pg)

        if params["use_binary_file"]:
            if not recording.binary_compatible_with(time_axis=0, file_paths_length=1):
                # local copy needed
                binary_file_path = sorter_output_folder / "recording.dat"
                write_binary_recording(
                    recording=recording,
                    file_paths=[binary_file_path],
                    **get_job_kwargs(params, verbose),
                )
                params["filename"] = str(binary_file_path)

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
        from kilosort.io import load_probe, RecordingExtractorAsArray, BinaryFiltered, save_preprocessing
        from kilosort.parameters import DEFAULT_SETTINGS

        import time
        import torch
        import numpy as np

        if verbose:
            import logging

            logging.basicConfig(level=logging.INFO)

        if version.parse(cls.get_sorter_version()) < version.parse("4.0.5"):
            raise RuntimeError(
                "Kilosort versions before 4.0.5 are not supported"
                "in SpikeInterface. "
                "Please upgrade Kilosort version."
            )

        sorter_output_folder = sorter_output_folder.absolute()

        probe_filename = sorter_output_folder / "probe.prb"

        torch_device = params["torch_device"]
        if torch_device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(torch_device)

        # load probe
        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        probe = load_probe(probe_path=probe_filename)
        probe_name = ""

        if params["use_binary_file"] is None:
            if recording.binary_compatible_with(time_axis=0, file_paths_length=1):
                # no copy
                binary_description = recording.get_binary_description()
                filename = str(binary_description["file_paths"][0])
                file_object = None
            else:
                # the recording is not binary compatible and no binary copy has been written.
                # in this case, we use the RecordingExtractorAsArray object
                filename = ""
                file_object = RecordingExtractorAsArray(recording_extractor=recording)
        elif params["use_binary_file"]:
            # here we force the use of a binary file
            if recording.binary_compatible_with(time_axis=0, file_paths_length=1):
                # no copy
                binary_description = recording.get_binary_description()
                filename = str(binary_description["file_paths"][0])
                file_object = None
            else:
                # a local copy has been written
                filename = str(sorter_output_folder / "recording.dat")
                file_object = None
        else:
            # here we force the use of the RecordingExtractorAsArray object
            filename = ""
            file_object = RecordingExtractorAsArray(recording_extractor=recording)

        data_dtype = recording.get_dtype()

        do_CAR = params["do_CAR"]
        invert_sign = params["invert_sign"]
        save_extra_vars = params["save_extra_vars"]
        save_preprocessed_copy = params["save_preprocessed_copy"]
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
        bad_channels = params["bad_channels"]
        clear_cache = params["clear_cache"]

        filename, data_dir, results_dir, probe = set_files(
            settings=settings,
            filename=filename,
            probe=probe,
            probe_name=probe_name,
            data_dir=data_dir,
            results_dir=results_dir,
            bad_channels=bad_channels,
        )

        ops = initialize_ops(
            settings=settings,
            probe=probe,
            data_dtype=data_dtype,
            do_CAR=do_CAR,
            invert_sign=invert_sign,
            device=device,
            save_preprocessed_copy=save_preprocessed_copy,
        )

        n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, _, _, tmin, tmax, artifact, _, _ = (
            get_run_parameters(ops)
        )

        # Set preprocessing and drift correction parameters
        if not params["skip_kilosort_preprocessing"]:
            ops = compute_preprocessing(ops=ops, device=device, tic0=tic0, file_object=file_object)
        else:
            print("Skipping kilosort preprocessing.")
            bfile = BinaryFiltered(
                filename=ops["filename"],
                n_chan_bin=n_chan_bin,
                fs=fs,
                NT=NT,
                nt=nt,
                nt0min=twav_min,
                chan_map=chan_map,
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
        #            bfile.close()  # TODO: KS do this after preprocessing?

        np.random.seed(1)
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1)

        if not params["do_correction"]:
            print("Skipping drift correction.")
            ops["nblocks"] = 0

        # this function applies both preprocessing and drift correction
        ops, bfile, st0 = compute_drift_correction(
            ops=ops,
            device=device,
            tic0=tic0,
            progress_bar=progress_bar,
            file_object=file_object,
            clear_cache=clear_cache,
        )

        if save_preprocessed_copy:
            save_preprocessing(results_dir / "temp_wh.dat", ops, bfile)

        # Sort spikes and save results
        st, tF, _, _ = detect_spikes(
            ops=ops, device=device, bfile=bfile, tic0=tic0, progress_bar=progress_bar, clear_cache=clear_cache
        )

        clu, Wall = cluster_spikes(
            st=st,
            tF=tF,
            ops=ops,
            device=device,
            bfile=bfile,
            tic0=tic0,
            progress_bar=progress_bar,
            clear_cache=clear_cache,
        )

        if params["skip_kilosort_preprocessing"]:
            ops["preprocessing"] = dict(
                hp_filter=torch.as_tensor(np.zeros(1)), whiten_mat=torch.as_tensor(np.eye(recording.get_num_channels()))
            )

        _ = save_sorting(
            ops=ops,
            results_dir=results_dir,
            st=st,
            clu=clu,
            tF=tF,
            Wall=Wall,
            imin=bfile.imin,
            tic0=tic0,
            save_extra_vars=save_extra_vars,
            save_preprocessed_copy=save_preprocessed_copy,
        )

        if params["delete_recording_dat"]:
            # only delete dat file if it was created by the wrapper
            if (sorter_output_folder / "recording.dat").is_file():
                (sorter_output_folder / "recording.dat").unlink()

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return KilosortBase._get_result_from_folder(sorter_output_folder)
