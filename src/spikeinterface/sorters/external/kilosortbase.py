from __future__ import annotations

from pathlib import Path
from warnings import warn
import json
import shutil
import sys

import numpy as np

from spikeinterface.sorters.utils import ShellScript, get_matlab_shell_name, get_bash_path
from spikeinterface.sorters.basesorter import get_job_kwargs
from spikeinterface.extractors import KiloSortSortingExtractor
from spikeinterface.core import write_binary_recording
from spikeinterface.preprocessing.zero_channel_pad import TracePaddedRecording


class KilosortBase:
    """
    Shared class for all kilosort implementation:
      * _generate_channel_map_file
      * _generate_ops_file
      * _run_from_folder
      * _get_result_from_folder
    """

    gpu_capability = "nvidia-required"
    requires_binary_data = True

    @staticmethod
    def _generate_channel_map_file(recording, sorter_output_folder):
        """
        This function generates channel map data for kilosort and saves as `chanMap.mat`

        Loading example in Matlab (shouldn't be assigned to a variable):
        >> load('/path/to/sorter_output_folder/chanMap.mat');

        Parameters
        ----------
        recording: BaseRecording
            The recording to generate the channel map file
        sorter_output_folder: pathlib.Path
            Path object to save `chanMap.mat` file
        """
        # prepare electrode positions for this group (only one group, the split is done in basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        nchan = recording.get_num_channels()
        xcoords = ([p[0] for p in positions],)
        ycoords = ([p[1] for p in positions],)
        kcoords = (groups,)

        channel_map = {}
        channel_map["Nchannels"] = nchan
        channel_map["connected"] = np.full((nchan, 1), True)
        channel_map["chanMap0ind"] = np.arange(nchan)
        channel_map["chanMap"] = channel_map["chanMap0ind"] + 1

        channel_map["xcoords"] = np.array(xcoords).astype(float)
        channel_map["ycoords"] = np.array(ycoords).astype(float)
        channel_map["kcoords"] = np.array(kcoords).astype(float)

        sample_rate = recording.get_sampling_frequency()
        channel_map["fs"] = float(sample_rate)
        import scipy.io

        scipy.io.savemat(str(sorter_output_folder / "chanMap.mat"), channel_map)

    @classmethod
    def _generate_ops_file(cls, recording, params, sorter_output_folder, binary_file_path):
        """
        This function generates ops (configs) data for kilosort and saves as `ops.mat`

        Loading example in Matlab (shouldn't be assigned to a variable):
        >> load('/sorter_output_folder/ops.mat');

        Parameters
        ----------
        recording : BaseRecording
            The recording to generate the channel map file
        params : dict
            Custom parameters dictionary for kilosort
        sorter_output_folder : pathlib.Path
            Path object to save `ops.mat`
        """
        ops = {}

        nchan = float(recording.get_num_channels())
        ops["NchanTOT"] = nchan  # total number of channels (omit if already in chanMap file)
        ops["Nchan"] = nchan  # number of active channels (omit if already in chanMap file)

        ops["datatype"] = "dat"  # binary ('dat', 'bin') or 'openEphys'
        ops["fbinary"] = str(binary_file_path.absolute())  # will be created for 'openEphys'
        ops["fproc"] = str((sorter_output_folder / "temp_wh.dat").absolute())  # residual from RAM of preprocessed data
        ops["root"] = str(sorter_output_folder.absolute())  # 'openEphys' only: where raw files are
        ops["trange"] = [0, np.inf]  #  time range to sort
        ops["chanMap"] = str((sorter_output_folder / "chanMap.mat").absolute())

        ops["fs"] = recording.get_sampling_frequency()  # sample rate
        ops["CAR"] = 1.0 if params["car"] else 0.0

        ops = cls._get_specific_options(ops, params)

        # Converting integer values into float
        # Kilosort interprets ops fields as double
        for k, v in ops.items():
            if isinstance(v, int):
                ops[k] = float(v)

        ops = {"ops": ops}
        import scipy.io

        scipy.io.savemat(str(sorter_output_folder / "ops.mat"), ops)

    @classmethod
    def _get_specific_options(cls, ops, params):
        """Specific options should be implemented in subclass"""
        return ops

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        cls._generate_channel_map_file(recording, sorter_output_folder)

        skip_kilosort_preprocessing = params.get("skip_kilosort_preprocessing", False)

        if (
            recording.binary_compatible_with(dtype="int16", time_axis=0, file_paths_length=1)
            and not skip_kilosort_preprocessing
        ):
            # no copy
            binary_description = recording.get_binary_description()
            binary_file_path = Path(binary_description["file_paths"][0])
        else:
            # local copy needed
            binary_file_path = sorter_output_folder / "recording.dat"
            if skip_kilosort_preprocessing:
                # when we skip the kilosort preprocessing we need to extend the file with zero pad to ensure
                # that the number of sample is compatible with batch size (NT)
                nt = params["NT"]
                num_samples = recording.get_num_samples()
                pad = nt - num_samples % nt

                padding_start = 0
                padding_end = pad
                padded_recording = TracePaddedRecording(
                    recording=recording,
                    padding_start=padding_start,
                    padding_end=padding_end,
                )
            else:
                padded_recording = recording
            write_binary_recording(
                recording=padded_recording,
                file_paths=[binary_file_path],
                dtype="int16",
                **get_job_kwargs(params, verbose),
            )

        cls._generate_ops_file(recording, params, sorter_output_folder, binary_file_path)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        sorter_output_folder = sorter_output_folder.absolute()
        if cls.check_compiled():
            shell_cmd = f"""
                #!/bin/bash
                {cls.compiled_name} "{sorter_output_folder}"
            """
        else:
            source_dir = Path(Path(__file__).parent)
            external_dir = source_dir.parent
            shutil.copy(str(source_dir / f"{cls.sorter_name}_master.m"), str(sorter_output_folder))
            shutil.copy(str(external_dir / "utils" / "writeNPY.m"), str(sorter_output_folder))
            shutil.copy(str(external_dir / "utils" / "constructNPYheader.m"), str(sorter_output_folder))

            sorter_path = getattr(cls, f"{cls.sorter_name}_path")
            sorter_path = Path(sorter_path).absolute()
            if "win" in sys.platform and sys.platform != "darwin":
                disk_move = str(sorter_output_folder)[:2]
                shell_cmd = f"""
                    {disk_move}
                    cd {sorter_output_folder}
                    matlab -nosplash -wait -r "{cls.sorter_name}_master('{sorter_output_folder}', '{sorter_path}')"
                """
            else:
                if get_matlab_shell_name() == "fish":
                    # Avoid MATLAB's 'copyfile' function failing due to MATLAB using fish as a shell
                    bash_path = get_bash_path()
                    warn(
                        f"Avoid Kilosort failing due to MATLAB using 'fish' as a shell: setting `MATLAB_SHELL` env variable to `{bash_path}`."
                    )
                    matlab_shell_str = f"""
                    export MATLAB_SHELL="{bash_path}"
                    echo "Set MATLAB shell to $MATLAB_SHELL"
                    """
                else:
                    matlab_shell_str = ""
                shell_cmd = f"""
                    #!/bin/bash
                    {matlab_shell_str}
                    cd "{sorter_output_folder}"
                    matlab -nosplash -nodisplay -r "{cls.sorter_name}_master('{sorter_output_folder}', '{sorter_path}')"
                """
        shell_script = ShellScript(
            shell_cmd,
            script_path=sorter_output_folder / f"run_{cls.sorter_name}",
            log_path=sorter_output_folder / f"{cls.sorter_name}.log",
            verbose=verbose,
        )
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception(f"{cls.sorter_name} returned a non-zero exit code")

        # Clean-up temporary files
        if params["delete_recording_dat"] and (recording_file := sorter_output_folder / "recording.dat").exists():
            recording_file.unlink()

        all_tmp_files = ("matlab_files", "temp_wh.dat")

        if isinstance(params["delete_tmp_files"], bool):
            if params["delete_tmp_files"]:
                tmp_files_to_remove = all_tmp_files
            else:
                tmp_files_to_remove = ()
        else:
            assert isinstance(
                params["delete_tmp_files"], (tuple, list)
            ), "`delete_tmp_files` must be a `Bool`, `Tuple` or `List`."

            for name in params["delete_tmp_files"]:
                assert name in all_tmp_files, f"{name} is not a valid option, must be one of: {all_tmp_files}"

            tmp_files_to_remove = params["delete_tmp_files"]

        if "temp_wh.dat" in tmp_files_to_remove:
            if (temp_wh_file := sorter_output_folder / "temp_wh.dat").exists():
                temp_wh_file.unlink()

        if "matlab_files" in tmp_files_to_remove:
            for ext in ["*.m", "*.mat"]:
                for temp_file in sorter_output_folder.glob(ext):
                    temp_file.unlink()

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
        sorting = KiloSortSortingExtractor(folder_path=sorter_output_folder, keep_good_only=keep_good_only)
        return sorting
