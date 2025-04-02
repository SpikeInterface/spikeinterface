from __future__ import annotations

import os
from pathlib import Path
from typing import Union
import sys

from spikeinterface.sorters.utils import ShellScript
from spikeinterface.sorters.basesorter import BaseSorter, get_job_kwargs

from spikeinterface.extractors import MdaRecordingExtractor, MdaSortingExtractor

PathType = Union[str, Path]


def check_if_installed(ironclust_path: Union[str, None]):
    if ironclust_path is None:
        return False
    assert isinstance(ironclust_path, str)

    if ironclust_path.startswith('"'):
        ironclust_path = ironclust_path[1:-1]
    ironclust_path = str(Path(ironclust_path).absolute())

    if (Path(ironclust_path) / "matlab" / "irc2.m").is_file():
        return True
    else:
        return False


class IronClustSorter(BaseSorter):
    """IronClust Sorter object."""

    sorter_name: str = "ironclust"
    compiled_name: str = "p_ironclust"
    ironclust_path: Union[str, None] = os.getenv("IRONCLUST_PATH", None)

    requires_locations = True
    gpu_capability = "nvidia-optional"
    requires_binary_data = True

    _default_params = {
        "detect_sign": -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        "adjacency_radius": 50,  # Use -1 to include all channels in every neighborhood
        "adjacency_radius_out": 100,  # Use -1 to include all channels in every neighborhood
        "detect_threshold": 3.5,  # detection threshold
        "prm_template_name": "",  # .prm template file name
        "freq_min": 300,
        "freq_max": 8000,
        "merge_thresh": 0.985,  # Threshold for automated merging
        "pc_per_chan": 9,  # Number of principal components per channel
        "whiten": False,  # Whether to do channel whitening as part of preprocessing
        "filter_type": "bandpass",  # none, bandpass, wiener, fftdiff, ndiff
        "filter_detect_type": "none",  # none, bandpass, wiener, fftdiff, ndiff
        "common_ref_type": "trimmean",  # none, mean, median
        "batch_sec_drift": 300,  # batch duration in seconds. clustering time duration
        "step_sec_drift": 20,  # compute anatomical similarity every n sec
        "knn": 30,  # K nearest neighbors
        "min_count": 30,  # Minimum cluster size
        "fGpu": True,  # Use GPU if available
        "fft_thresh": 8,  # FFT-based noise peak threshold
        "fft_thresh_low": 0,  # FFT-based noise peak lower threshold (set to 0 to disable dual thresholding scheme)
        "nSites_whiten": 16,  # Number of adjacent channels to whiten
        "feature_type": "gpca",  # gpca, pca, vpp, vmin, vminmax, cov, energy, xcov
        "delta_cut": 1,  # Cluster detection threshold (delta-cutoff)
        "post_merge_mode": 1,  # post merge mode
        "sort_mode": 1,  # sort mode
        "fParfor": False,  # parfor loop
        "filter": True,  # Enable or disable filter
        "clip_pre": 0.25,  # pre-peak clip duration in ms
        "clip_post": 0.75,  # post-peak clip duration in ms
        "merge_thresh_cc": 1,  # cross-correlogram merging threshold, set to 1 to disable
        "nRepeat_merge": 3,  # number of repeats for merge
        "merge_overlap_thresh": 0.95,  # knn-overlap merge threshold
        "version": 2,
    }

    _params_description = {
        "detect_sign": "Use -1 (negative), 1 (positive) or 0 (both) depending "
        "on the sign of the spikes in the recording",
        "adjacency_radius": "Use -1 to include all channels in every neighborhood",
        "adjacency_radius_out": "Use -1 to include all channels in every neighborhood",
        "detect_threshold": "detection threshold",
        "prm_template_name": ".prm template file name",
        "freq_min": "High-pass filter cutoff frequency",
        "freq_max": "Low-pass filter cutoff frequency",
        "merge_thresh": "Threshold for automated merging",
        "pc_per_chan": "Number of principal components per channel",
        "whiten": "Whether to do channel whitening as part of preprocessing",
        "filter_type": "Filter type: none, bandpass, wiener, fftdiff, ndiff",
        "filter_detect_type": "Filter type for detection: none, bandpass, wiener, fftdiff, ndiff",
        "common_ref_type": "Common reference type: none, mean, median, trimmean",
        "batch_sec_drift": "Batch duration in seconds. clustering time duration",
        "step_sec_drift": "Compute anatomical similarity every n sec",
        "knn": "K nearest neighbors",
        "min_count": "Minimum cluster size",
        "fGpu": "Use GPU if True",
        "fft_thresh": "FFT-based noise peak threshold",
        "fft_thresh_low": "FFT-based noise peak lower threshold (set to 0 to disable dual thresholding scheme)",
        "nSites_whiten": "Number of adjacent channels to whiten",
        "feature_type": "gpca, pca, vpp, vmin, vminmax, cov, energy, xcov",
        "delta_cut": "Cluster detection threshold (delta-cutoff)",
        "post_merge_mode": "Post merge mode",
        "sort_mode": "Sort mode",
        "fParfor": "Parfor loop",
        "filter": "Enable or disable filter",
        "clip_pre": "Pre-peak clip duration in ms",
        "clip_post": "Post-peak clip duration in ms",
        "merge_thresh_cc": "Cross-correlogram merging threshold, set to 1 to disable",
        "nRepeat_merge": "Number of repeats for merge",
        "merge_overlap_thresh": "Knn-overlap merge threshold",
        "version": "The irc command version. Can be 1 or 2 (default)",
    }

    sorter_descrpition = """Ironclust is a density-based spike sorter designed for high-density probes
    (e.g. Neuropixels). It uses features and spike location estimates for clustering, and it performs a drift
    correction. For more information see https://doi.org/10.1101/101030"""

    installation_mesg = """\nTo use IronClust run:\n
        >>> git clone https://github.com/flatironinstitute/ironclust
    and provide the installation path by setting the IRONCLUST_PATH
    environment variables or using IronClustSorter.set_ironclust_path().\n\n
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        if cls.check_compiled():
            return True
        return check_if_installed(cls.ironclust_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return "compiled"
        version_filename = Path(os.environ["IRONCLUST_PATH"]) / "matlab" / "version.txt"
        if version_filename.is_file():
            with open(str(version_filename), mode="r", encoding="utf8") as f:
                line = f.readline()
                d = {}
                exec(line, None, d)
                version = d["version"]
                return version
        return "unknown"

    @classmethod
    def use_gpu(cls, params):
        if "fGpu" in params:
            return params["fGpu"]
        return cls.default_params()["fGpu"]

    @staticmethod
    def set_ironclust_path(ironclust_path: PathType):
        ironclust_path = str(Path(ironclust_path).absolute())
        IronClustSorter.ironclust_path = ironclust_path
        try:
            print("Setting IRONCLUST_PATH environment variable for subprocess calls to:", ironclust_path)
            os.environ["IRONCLUST_PATH"] = ironclust_path
        except Exception as e:
            print("Could not set IRONCLUST_PATH environment variable:", e)

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params["filter"]

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        p = params

        dataset_dir = sorter_output_folder / "ironclust_dataset"
        # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
        MdaRecordingExtractor.write_recording(
            recording=recording, save_path=str(dataset_dir), verbose=False, **get_job_kwargs(params, verbose)
        )

        samplerate = recording.get_sampling_frequency()
        num_channels = recording.get_num_channels()
        num_timepoints = recording.get_num_frames(segment_index=0)
        duration_minutes = num_timepoints / samplerate / 60
        if verbose:
            print(f"channels = {num_channels}, timepoints = {num_timepoints}, duration = {duration_minutes} minutes")

        if verbose:
            print("Creating argfile.txt..")
        txt = ""
        for key0, val0 in params.items():
            txt += "{}={}\n".format(key0, val0)
        txt += "samplerate={}\n".format(samplerate)
        with (dataset_dir / "argfile.txt").open("w") as f:
            f.write(txt)

        # TODO remove this because recording.json contain the sample_rate natively
        tmpdir = sorter_output_folder / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        samplerate_fname = str(tmpdir / "samplerate.txt")
        with open(samplerate_fname, "w") as f:
            f.write("{}".format(samplerate))

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        dataset_dir = (sorter_output_folder / "ironclust_dataset").absolute()
        source_dir = (Path(__file__).parent).absolute()

        tmpdir = (sorter_output_folder / "tmp").absolute()

        if verbose:
            print("Running ironclust in {tmpdir}..".format(tmpdir=str(tmpdir)))

        if cls.check_compiled():
            shell_cmd = """
                #!/bin/bash
                p_ironclust {tmpdir} {dataset_dir}/raw.mda {dataset_dir}/geom.csv '' '' {tmpdir}/firings.mda {dataset_dir}/argfile.txt
            """.format(
                tmpdir=str(tmpdir), dataset_dir=str(dataset_dir)
            )
        else:
            cmd = """
                addpath('{source_dir}');
                addpath('{ironclust_path}', '{ironclust_path}/matlab', '{ironclust_path}/matlab/mdaio');
                try
                    p_ironclust('{tmpdir}', '{dataset_dir}/raw.mda', '{dataset_dir}/geom.csv', '', '', '{tmpdir}/firings.mda', '{dataset_dir}/argfile.txt');
                catch
                    fprintf('----------------------------------------');
                    fprintf(lasterr());
                    quit(1);
                end
                quit(0);
            """
            cmd = cmd.format(
                ironclust_path=IronClustSorter.ironclust_path,
                tmpdir=str(tmpdir),
                dataset_dir=str(dataset_dir),
                source_dir=str(source_dir),
            )

            matlab_cmd = ShellScript(cmd, script_path=str(tmpdir / "run_ironclust.m"))
            matlab_cmd.write()

            if "win" in sys.platform and sys.platform != "darwin":
                shell_cmd = """
                    {disk_move}
                    cd {tmpdir}
                    matlab -nosplash -wait -log -r run_ironclust
                """.format(
                    disk_move=str(tmpdir)[:2], tmpdir=tmpdir
                )
            else:
                shell_cmd = """
                    #!/bin/bash
                    cd "{tmpdir}"
                    matlab -nosplash -nodisplay -log -r run_ironclust
                """.format(
                    tmpdir=tmpdir
                )

        shell_script = ShellScript(
            shell_cmd,
            script_path=sorter_output_folder / f"run_{cls.sorter_name}",
            log_path=sorter_output_folder / f"{cls.sorter_name}.log",
            verbose=verbose,
        )
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception("ironclust returned a non-zero exit code")

        result_fname = tmpdir / "firings.mda"
        if not result_fname.is_file():
            raise Exception(f"Result file does not exist: {result_fname}")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        tmpdir = sorter_output_folder / "tmp"

        result_fname = str(tmpdir / "firings.mda")
        samplerate_fname = str(tmpdir / "samplerate.txt")
        with open(samplerate_fname, "r") as f:
            samplerate = float(f.read())

        sorting = MdaSortingExtractor(file_path=result_fname, sampling_frequency=samplerate)

        return sorting
