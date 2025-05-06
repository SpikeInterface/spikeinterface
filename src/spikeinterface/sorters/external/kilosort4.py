from __future__ import annotations

from pathlib import Path
from typing import Union
import warnings
from packaging import version


from spikeinterface.core import write_binary_recording
from spikeinterface.sorters.basesorter import BaseSorter, get_job_kwargs
from .kilosortbase import KilosortBase
from spikeinterface.sorters.basesorter import get_job_kwargs
from importlib.metadata import version as importlib_version

PathType = Union[str, Path]


class Kilosort4Sorter(BaseSorter):
    """Kilosort4 Sorter object."""

    sorter_name: str = "kilosort4"
    requires_locations = True
    gpu_capability = "nvidia-optional"
    requires_binary_data = True

    _si_default_params = {
        "do_CAR": True,
        "invert_sign": False,
        "save_extra_vars": False,
        "save_preprocessed_copy": False,
        "torch_device": "auto",
        "bad_channels": None,
        "clear_cache": False,
        "do_correction": True,
        "skip_kilosort_preprocessing": False,
        "keep_good_only": False,
        "use_binary_file": True,
        "delete_recording_dat": True,
    }

    _si_params_description = {
        "do_CAR": "If True, common average reference is performed. Default is True. (run_kilosrt parameter)",
        "invert_sign": "Invert the sign of the data. Default value: False. (run_kilosort parameter)",
        "save_extra_vars": "If True, additional kwargs are saved to the output. Default is False. (run_kilosort parameter)",
        "save_preprocessed_copy": "Save a pre-processed copy of the data (including drift correction) to temp_wh.dat in the results directory and format Phy output to use that copy of the data. (run_kilosort parameter)",
        "torch_device": "Select the torch device auto/cuda/cpu. Default is 'auto'. (run_kilosort parameter)",
        "bad_channels": "A list of channel indices (rows in the binary file) that should not be included in sorting. Listing channels here is equivalent to excluding them from the probe dictionary. (run_kilosort parameter)",
        "clear_cache": "If True, force pytorch to free up memory reserved for its cache in between memory-intensive operations. Note that setting `clear_cache=True` is NOT recommended unless you encounter GPU out-of-memory errors, since this can result in slower sorting. (run_kilosort parameter)",
        "do_correction": "If True, drift correction is performed. Default is True. (spikeinterface parameter)",
        "skip_kilosort_preprocessing": "Can optionally skip the internal kilosort preprocessing. (spikeinterface parameter)",
        "keep_good_only": "If True, only the units labeled as 'good' by Kilosort are returned in the output. (spikeinterface parameter)",
        "use_binary_file": "If True then Kilosort is run using a binary file. In this case, if the input recording is not binary compatible, it is written to a binary file in the output folder. "
        "If False then Kilosort is run on the recording object directly using the RecordingExtractorAsArray object. If None, then if the recording is binary compatible, the sorter will use the binary file, otherwise the RecordingExtractorAsArray. "
        "Default is True. (spikeinterface parameter)",
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
        import importlib.util

        ks_spec = importlib.util.find_spec("kilosort")
        torch_spec = importlib.util.find_spec("torch")
        if ks_spec is not None and torch_spec is not None:
            HAVE_KS = True
        else:
            HAVE_KS = False
        return HAVE_KS

    @classmethod
    def get_sorter_version(cls):
        """kilosort.__version__ <4.0.10 is always '4'"""
        return importlib_version("kilosort")

    @classmethod
    def _dynamic_params(cls):
        if cls.is_installed():
            import kilosort as ks

            # we skip some parameters that are not relevant for the user
            # n_chan_bin/sampling_frequency: retrieved from the recording
            # tmin/tmax: same ase time/frame_slice in SpikeInterface
            skip_main = ["n_chan_bin", "sampling_frequency", "tmin", "tmax"]
            default_params = {}
            default_params_descriptions = {}
            ks_params = ks.parameters.MAIN_PARAMETERS.copy()
            ks_params.update(ks.parameters.EXTRA_PARAMETERS)
            for param, param_value in ks_params.items():
                if param not in skip_main:
                    default_params[param] = param_value["default"]
                    desc = param_value.get("description")
                    if desc is None:
                        desc = ""
                    else:
                        # get rid of escape characters and extra spaces
                        desc = " ".join(desc.replace("\n", "").split())
                    default_params_descriptions[param] = desc
            default_params.update(cls._si_default_params)
            default_params_descriptions.update(cls._si_params_description)
            return default_params, default_params_descriptions
        else:
            warnings.warn("Kilosort4 is not installed. Please install kilosort4 to get the parameters.")
            return {}, {}

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
        cls._setup_json_probe_map(recording, sorter_output_folder)

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
        from kilosort import __version__ as ks_version
        from kilosort.run_kilosort import (
            set_files,
            initialize_ops,
            compute_preprocessing,
            compute_drift_correction,
            detect_spikes,
            cluster_spikes,
            save_sorting,
            get_run_parameters,
            setup_logger,
        )
        from kilosort.io import load_probe, RecordingExtractorAsArray, BinaryFiltered, save_preprocessing
        from kilosort.parameters import DEFAULT_SETTINGS

        import time
        import torch
        import numpy as np
        import logging

        if version.parse(cls.get_sorter_version()) < version.parse("4.0.16"):
            raise RuntimeError(
                "Kilosort versions before 4.0.16 are not supported"
                "in SpikeInterface. "
                "Please upgrade Kilosort version."
            )

        # setup kilosort's console and file log handlers
        setup_logger_takes_verbose_console = version.parse(cls.get_sorter_version()) > version.parse("4.0.18")
        logger_is_named = version.parse(cls.get_sorter_version()) > version.parse("4.0.20")

        if setup_logger_takes_verbose_console:
            # v4.0.19 and higher
            setup_logger(sorter_output_folder, verbose_console=False)
        else:
            # v4.0.16, v4.0.17, v4.0.18
            setup_logger(sorter_output_folder)

        # if verbose is False, set the stream handler's log
        # level to logging.WARNING to preserve original
        # behavior prior to addition of setup_logger() above
        if not verbose:
            if logger_is_named:
                # v4.0.21 and above
                logger = logging.getLogger("kilosort")
            else:
                # v4.0.16, v4.0.17, v4.0.18, v4.0.19, v4.0.20
                logger = logging.getLogger("")

            # find the stream handler
            stream_handler = None
            for handler in logger.handlers:
                if type(handler) == logging.StreamHandler:
                    stream_handler = handler
                    break

            stream_handler.setLevel(logging.WARNING)

        sorter_output_folder = sorter_output_folder.absolute()

        probe_filename = sorter_output_folder / "chanMap.json"

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
            if verbose:
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

        set_files_kwargs = dict(
            settings=settings,
            filename=filename,
            probe=probe,
            probe_name=probe_name,
            data_dir=data_dir,
            results_dir=results_dir,
            bad_channels=bad_channels,
        )
        if version.parse(ks_version) >= version.parse("4.0.34"):
            set_files_kwargs.update(dict(shank_idx=None))

        filename, data_dir, results_dir, probe = set_files(**set_files_kwargs)

        ops = initialize_ops(
            settings=settings,
            probe=probe,
            data_dtype=data_dtype,
            do_CAR=do_CAR,
            invert_sign=invert_sign,
            device=device,
            save_preprocessed_copy=save_preprocessed_copy,
        )
        if version.parse(ks_version) >= version.parse("4.0.34"):
            ops = ops[0]

        n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, _, _, tmin, tmax, artifact, _, _ = (
            get_run_parameters(ops)
        )

        # Set preprocessing and drift correction parameters
        if not params["skip_kilosort_preprocessing"]:
            ops = compute_preprocessing(ops=ops, device=device, tic0=tic0, file_object=file_object)
        else:
            if verbose:
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
            if verbose:
                print("Skipping drift correction.")
            ops["nblocks"] = 0

        drift_kwargs = dict(
            ops=ops,
            device=device,
            tic0=tic0,
            progress_bar=progress_bar,
            file_object=file_object,
            clear_cache=clear_cache,
        )
        if version.parse(ks_version) >= version.parse("4.0.28"):
            drift_kwargs.update(dict(verbose=verbose))

        # this function applies both preprocessing and drift correction
        ops, bfile, st0 = compute_drift_correction(**drift_kwargs)

        if save_preprocessed_copy:
            save_preprocessing(results_dir / "temp_wh.dat", ops, bfile)

        # Sort spikes and save results
        detect_spikes_kwargs = dict(
            ops=ops,
            device=device,
            bfile=bfile,
            tic0=tic0,
            progress_bar=progress_bar,
            clear_cache=clear_cache,
        )
        if version.parse(ks_version) >= version.parse("4.0.28"):
            detect_spikes_kwargs.update(dict(verbose=verbose))
        st, tF, _, _ = detect_spikes(**detect_spikes_kwargs)

        cluster_spikes_kwargs = dict(
            st=st,
            tF=tF,
            ops=ops,
            device=device,
            bfile=bfile,
            tic0=tic0,
            progress_bar=progress_bar,
            clear_cache=clear_cache,
        )
        if version.parse(ks_version) >= version.parse("4.0.28"):
            cluster_spikes_kwargs.update(dict(verbose=verbose))
        if version.parse(ks_version) <= version.parse("4.0.30"):
            clu, Wall = cluster_spikes(**cluster_spikes_kwargs)
        else:
            clu, Wall, st, tF = cluster_spikes(**cluster_spikes_kwargs)

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

    @classmethod
    def _setup_json_probe_map(cls, recording, sorter_output_folder):
        """Create a JSON probe map file for Kilosort4."""
        from kilosort.io import save_probe
        import numpy as np

        groups = recording.get_channel_groups()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead.")

        n_chan = recording.get_num_channels()
        chanMap = np.arange(n_chan)
        xc = positions[:, 0]
        yc = positions[:, 1]
        kcoords = groups.astype(float)

        probe = {
            "chanMap": chanMap,
            "xc": xc,
            "yc": yc,
            "kcoords": kcoords,
            "n_chan": n_chan,
        }
        save_probe(probe, str(sorter_output_folder / "chanMap.json"))
