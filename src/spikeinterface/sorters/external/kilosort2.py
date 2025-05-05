from __future__ import annotations

from pathlib import Path
import os
from typing import Union

from spikeinterface.sorters.basesorter import BaseSorter
from .kilosortbase import KilosortBase
from spikeinterface.sorters.utils import get_git_commit

PathType = Union[str, Path]


def check_if_installed(kilosort2_path: Union[str, None]):
    if kilosort2_path is None:
        return False
    assert isinstance(kilosort2_path, str)

    if kilosort2_path.startswith('"'):
        kilosort2_path = kilosort2_path[1:-1]
    kilosort2_path = str(Path(kilosort2_path).absolute())
    if (Path(kilosort2_path) / "master_kilosort.m").is_file() or (Path(kilosort2_path) / "main_kilosort.m").is_file():
        return True
    else:
        return False


class Kilosort2Sorter(KilosortBase, BaseSorter):
    """Kilosort2 Sorter object."""

    sorter_name: str = "kilosort2"
    compiled_name: str = "ks2_compiled"
    kilosort2_path: Union[str, None] = os.getenv("KILOSORT2_PATH", None)
    requires_locations = False

    _default_params = {
        "detect_threshold": 6,
        "projection_threshold": [10, 4],
        "preclust_threshold": 8,
        "whiteningRange": 32,  # samples of the template to use for whitening "spatial" dimension
        "momentum": [20.0, 400.0],
        "car": True,
        "minFR": 0.1,
        "minfr_goodchannels": 0.1,
        "freq_min": 150,
        "sigmaMask": 30,
        "lam": 10.0,
        "nPCs": 3,
        "ntbuff": 64,
        "nfilt_factor": 4,
        "NT": None,
        "AUCsplit": 0.9,
        "wave_length": 61,
        "keep_good_only": False,
        "skip_kilosort_preprocessing": False,
        "scaleproc": None,
        "save_rez_to_mat": False,
        "delete_tmp_files": ("matlab_files",),
        "delete_recording_dat": False,
    }

    _params_description = {
        "detect_threshold": "Threshold for spike detection",
        "projection_threshold": "Threshold on projections",
        "preclust_threshold": "Threshold crossings for pre-clustering (in PCA projection space)",
        "whiteningRange": "Number of channels to use for whitening each channel",
        "momentum": "Number of samples to average over (annealed from first to second value)",
        "car": "Enable or disable common reference",
        "minFR": "Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed",
        "minfr_goodchannels": "Minimum firing rate on a 'good' channel",
        "freq_min": "High-pass filter cutoff frequency",
        "sigmaMask": "Spatial constant in um for computing residual variance of spike",
        "lam": "The importance of the amplitude penalty (like in Kilosort1: 0 means not used, 10 is average, 50 is a lot)",
        "nPCs": "Number of PCA dimensions",
        "ntbuff": "Samples of symmetrical buffer for whitening and spike detection",
        "nfilt_factor": "Max number of clusters per good channel (even temporary ones) 4",
        "NT": "Batch size (if None it is automatically computed--recommended Kilosort behavior if ntbuff also not changed)",
        "AUCsplit": "Threshold on the area under the curve (AUC) criterion for performing a split in the final step",
        "wave_length": "size of the waveform extracted around each detected peak, (Default 61, maximum 81)",
        "keep_good_only": "If True only 'good' units are returned",
        "skip_kilosort_preprocessing": "Can optionally skip the internal kilosort preprocessing",
        "scaleproc": "int16 scaling of whitened data, if None set to 200.",
        "save_rez_to_mat": "Save the full rez internal struc to mat file",
        "delete_tmp_files": "Delete temporary files created during sorting (matlab files and the `temp_wh.dat` file that "
        "contains kilosort-preprocessed data). Accepts `False` (deletes no files), `True` (deletes all files) "
        "or a Tuple containing the files to delete. Options are: ('temp_wh.dat', 'matlab_files')",
        "delete_recording_dat": "Whether to delete the 'recording.dat' file after a successful run",
    }

    sorter_description = """Kilosort2 is a GPU-accelerated and efficient template-matching spike sorter. On top of its
    predecessor Kilosort, it implements a drift-correction strategy.
    For more information see https://github.com/MouseLand/Kilosort2"""

    installation_mesg = """\nTo use Kilosort2 run:\n
        >>> git clone https://github.com/MouseLand/Kilosort2
    and provide the installation path by setting the KILOSORT2_PATH
    environment variables or using Kilosort2Sorter.set_kilosort2_path().\n\n

    More information on Kilosort2 at:
        https://github.com/MouseLand/Kilosort2
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        if cls.check_compiled():
            return True
        return check_if_installed(cls.kilosort2_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return "compiled"
        commit = get_git_commit(os.getenv("KILOSORT2_PATH", None))
        if commit is None:
            return "unknown"
        else:
            return "git-" + commit

    @classmethod
    def set_kilosort2_path(cls, kilosort2_path: PathType):
        kilosort2_path = str(Path(kilosort2_path).absolute())
        Kilosort2Sorter.kilosort2_path = kilosort2_path
        try:
            print("Setting KILOSORT2_PATH environment variable for subprocess calls to:", kilosort2_path)
            os.environ["KILOSORT2_PATH"] = kilosort2_path
        except Exception as e:
            print("Could not set KILOSORT2_PATH environment variable:", e)

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        p = params
        if p["NT"] is None:
            p["NT"] = 64 * 1024 + p["ntbuff"]
        else:
            p["NT"] = p["NT"] // 32 * 32  # make sure is multiple of 32
        if p["wave_length"] % 2 != 1:
            p["wave_length"] = p["wave_length"] + 1  # The wave_length must be odd
        if p["wave_length"] > 81:
            p["wave_length"] = 81  # The wave_length must be <=81
        return p

    @classmethod
    def _get_specific_options(cls, ops, params) -> dict:
        """
        Adds specific options for Kilosort2 in the ops dict and returns the final dict

        Parameters
        ----------
        ops : dict
            options data
        params : dict
            Custom parameters dictionary for kilosort3

        Returns
        ----------
        ops : dict
            Final ops data
        """

        # frequency for high pass filtering (150)
        ops["fshigh"] = params["freq_min"]

        # minimum firing rate on a "good" channel (0 to skip)
        ops["minfr_goodchannels"] = params["minfr_goodchannels"]

        projection_threshold = [float(pt) for pt in params["projection_threshold"]]
        # threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops["Th"] = projection_threshold

        # how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops["lam"] = params["lam"]

        # splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
        ops["AUCsplit"] = params["AUCsplit"]

        # minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        ops["minFR"] = params["minFR"]

        momentum = [float(mom) for mom in params["momentum"]]
        # number of samples to average over (annealed from first to second value)
        ops["momentum"] = momentum

        # spatial constant in um for computing residual variance of spike
        ops["sigmaMask"] = params["sigmaMask"]

        # threshold crossings for pre-clustering (in PCA projection space)
        ops["ThPre"] = params["preclust_threshold"]

        ## danger, changing these settings can lead to fatal errors
        # options for determining PCs
        ops["spkTh"] = -params["detect_threshold"]  # spike threshold in standard deviations (-6)
        ops["reorder"] = 1.0  # whether to reorder batches for drift correction.
        ops["nskip"] = 25.0  # how many batches to skip for determining spike PCs

        ops["GPU"] = 1.0  # has to be 1, no CPU version yet, sorry
        # ops['Nfilt'] = 1024 # max number of clusters
        ops["nfilt_factor"] = params["nfilt_factor"]  # max number of clusters per good channel (even temporary ones)
        ops["ntbuff"] = params["ntbuff"]  # samples of symmetrical buffer for whitening and spike detection
        ops["NT"] = params[
            "NT"
        ]  # must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
        ops["whiteningRange"] = params["whiteningRange"]  # number of channels to use for whitening each channel
        ops["nSkipCov"] = 25.0  # compute whitening matrix from every N-th batch
        ops["nPCs"] = params["nPCs"]  # how many PCs to project the spikes into
        ops["useRAM"] = 0.0  # not yet available

        ## option for wavelength
        ops["nt0"] = params[
            "wave_length"
        ]  # size of the waveform extracted around each detected peak. Be sure to make it odd to make alignment easier.

        ops["skip_kilosort_preprocessing"] = params["skip_kilosort_preprocessing"]
        if params["skip_kilosort_preprocessing"]:
            ops["fproc"] = ops["fbinary"]
            assert (
                params["scaleproc"] is not None
            ), "When skip_kilosort_preprocessing=True scaleproc must explicitly given"

        # int16 scaling of whitened data, when None then scaleproc is set to 200.
        scaleproc = params["scaleproc"]
        ops["scaleproc"] = scaleproc if scaleproc is not None else 200.0

        ops["save_rez_to_mat"] = params["save_rez_to_mat"]

        return ops
